"""
train_sft.py — Supervised fine-tuning of DistilGPT2 on GSM8K CoT with LoRA (PEFT).

Key choices (all defensible in interview):
  * LoRA adapters on GPT2 attention projections (c_attn, c_proj) -> ~0.5-1% trainable params.
  * Prompt tokens are MASKED in the loss (labels=-100) so the model is only trained to
    PRODUCE the answer, not to memorize/echo the question.
  * Causal LM objective on the answer span only.

Saves the LoRA adapter to checkpoints/sft/.

Run (T4 defaults are sane):
  python src/train_sft.py
  python src/train_sft.py --epochs 3 --batch-size 16 --max-train 5000
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "distilgpt2"
MAX_LEN = 512  # GSM8K Q+A fits comfortably; distilgpt2 ctx is 1024.


class SFTDataset(Dataset):
    """Builds prompt+answer sequences with the prompt portion masked out of the loss."""

    def __init__(self, jsonl_path, tokenizer, max_len=MAX_LEN):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = [json.loads(l) for l in open(jsonl_path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(r["answer"], add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tok.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def make_collate(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "sft_train.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "checkpoints" / "sft"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # GPT2 has no pad token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tok.eos_token_id

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["c_attn", "c_proj"],  # GPT2 attention Conv1D layers
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    ds = SFTDataset(args.data, tok)
    if args.max_train:
        ds.rows = ds.rows[: args.max_train]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=make_collate(tok.pad_token_id))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    step = 0
    for ep in range(args.epochs):
        running = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{total_steps} loss {running/50:.4f}")
                running = 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved SFT LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
