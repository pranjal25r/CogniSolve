"""
train_dpo.py — Direct Preference Optimization, implemented from scratch in PyTorch.

Design that makes this both correct and T4-friendly:
  * Start from the SFT model (SFT LoRA merged into the base weights) -> this IS the
    DPO reference policy.
  * Add a FRESH LoRA adapter on top; that adapter is the trainable DPO policy.
  * Reference log-probs = forward pass with the adapter DISABLED (PEFT context manager),
    so we never hold a second model copy in memory.

DPO loss (Rafailov et al., 2023):
    pi_logratio  = logp_policy(chosen)    - logp_policy(rejected)
    ref_logratio = logp_ref(chosen)       - logp_ref(rejected)
    loss = -log_sigmoid( beta * (pi_logratio - ref_logratio) )

Saves the DPO LoRA adapter to checkpoints/dpo/.

Run:
  python src/train_dpo.py --epochs 2 --beta 0.1
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "distilgpt2"
MAX_LEN = 512


class PrefDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = [json.loads(l) for l in open(jsonl_path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def encode_pair(tok, prompt, response, max_len=MAX_LEN):
    """Return input_ids and a response-only mask (1 on response tokens, 0 elsewhere)."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + r_ids)[:max_len]
    mask = ([0] * len(p_ids) + [1] * len(r_ids))[:max_len]
    return ids, mask


def make_collate(tok):
    def collate(batch):
        seqs, masks = [], []
        for ex in batch:
            for key in ("chosen", "rejected"):
                ids, m = encode_pair(tok, ex["prompt"], ex[key])
                seqs.append(ids)
                masks.append(m)
        maxlen = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        input_ids, attn, resp_mask = [], [], []
        for s, m in zip(seqs, masks):
            pad = maxlen - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            resp_mask.append(m + [0] * pad)
        # rows are interleaved [chosen0, rejected0, chosen1, rejected1, ...]
        return (
            torch.tensor(input_ids),
            torch.tensor(attn),
            torch.tensor(resp_mask, dtype=torch.float),
        )
    return collate


def sequence_logp(model, input_ids, attn, resp_mask):
    """Sum of per-token log-probs over response tokens only. Returns [B]."""
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = resp_mask[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(PROJECT_ROOT / "checkpoints" / "sft"))
    ap.add_argument("--prefs", default=str(PROJECT_ROOT / "data" / "prefs.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "checkpoints" / "dpo"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8, help="number of PAIRS per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, PeftModel, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.pad_token = tok.eos_token

    # 1) Bake the SFT adapter into the base weights -> this is the reference policy.
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    sft_merged = PeftModel.from_pretrained(base, args.sft).merge_and_unload()
    sft_merged.config.pad_token_id = tok.eos_token_id

    # 2) Add a fresh LoRA adapter on top -> the trainable DPO policy.
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", target_modules=["c_attn", "c_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(sft_merged, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    ds = PrefDataset(args.prefs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate(tok))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    model.train()
    step = 0
    for ep in range(args.epochs):
        for input_ids, attn, resp_mask in dl:
            input_ids, attn, resp_mask = input_ids.to(device), attn.to(device), resp_mask.to(device)

            # Reference log-probs: adapter disabled, no grad.
            with torch.no_grad(), model.disable_adapter():
                ref_logp = sequence_logp(model, input_ids, attn, resp_mask)
            # Policy log-probs: adapter enabled, with grad.
            pol_logp = sequence_logp(model, input_ids, attn, resp_mask)

            # rows interleave chosen (even idx) / rejected (odd idx)
            pol_chosen, pol_rej = pol_logp[0::2], pol_logp[1::2]
            ref_chosen, ref_rej = ref_logp[0::2], ref_logp[1::2]

            pi_logratio = pol_chosen - pol_rej
            ref_logratio = ref_chosen - ref_rej
            loss = -F.logsigmoid(args.beta * (pi_logratio - ref_logratio)).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()

            step += 1
            if step % 20 == 0:
                acc = (pi_logratio > ref_logratio).float().mean().item()  # pref accuracy
                print(f"epoch {ep} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} pref_acc {acc:.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved DPO LoRA adapter -> {out_dir}")
    print("NOTE: at eval time, merge SFT first, then load this DPO adapter on top.")


if __name__ == "__main__":
    main()
