"""
evaluate.py — Pass@1, Pass@K and maj@K on GSM8K test for base / SFT / DPO models.

Metric definitions (state these in interviews):
  * Pass@1  : single greedy decode; correct if the extracted final answer matches gold.
  * Pass@K  : draw K sampled completions; correct if ANY is right (covers reasoning diversity).
  * maj@K   : self-consistency — take the majority-voted final answer across K samples.

Writes results/metrics.json and prints a comparison table.

Run:
  python src/evaluate.py --models base sft dpo --num-test 200 --k 4
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "distilgpt2"
PROMPT_TEMPLATE = "Question: {q}\nAnswer:"  # MUST match prepare_data.py


def extract_pred(text: str) -> str:
    # Strict: require the model to emit the final-answer marker.
    if "####" not in text:
        return ""                       # no marker => no valid answer
    tail = text.split("####")[-1].replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+\.?\d*", tail)
    return nums[0] if nums else ""       # answer is the number right after ####


def is_correct(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


def build_model(which, sft_dir, dpo_dir, tok, device):
    """which in {base, sft, dpo} -> a ready-to-generate merged model."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if which == "base":
        model = base
    elif which == "sft":
        model = PeftModel.from_pretrained(base, sft_dir).merge_and_unload()
    elif which == "dpo":
        sft_merged = PeftModel.from_pretrained(base, sft_dir).merge_and_unload()
        model = PeftModel.from_pretrained(sft_merged, dpo_dir).merge_and_unload()
    else:
        raise ValueError(which)
    model.config.pad_token_id = tok.eos_token_id
    return model.to(device).eval()


@torch.no_grad()
def generate(model, tok, prompts, device, k, sample, max_new_tokens, temperature, top_p):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    plen = enc["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if sample:
        kwargs.update(do_sample=True, num_return_sequences=k,
                      temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False, num_return_sequences=1)
    gen = model.generate(**enc, **kwargs)
    texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
    n_ret = k if sample else 1
    # group per prompt
    return [texts[i * n_ret:(i + 1) * n_ret] for i in range(len(prompts))]


def evaluate_model(which, rows, sft_dir, dpo_dir, tok, device, args):
    model = build_model(which, sft_dir, dpo_dir, tok, device)
    n = len(rows)
    pass1 = passk = majk = 0

    bs = args.eval_batch_size
    for s in range(0, n, bs):
        batch = rows[s:s + bs]
        prompts = [r["prompt"] for r in batch]
        golds = [r["gold"] for r in batch]

        greedy = generate(model, tok, prompts, device, 1, False,
                          args.max_new_tokens, args.temperature, args.top_p)
        sampled = generate(model, tok, prompts, device, args.k, True,
                           args.max_new_tokens, args.temperature, args.top_p)

        for gd, sm, gold in zip(greedy, sampled, golds):
            if is_correct(extract_pred(gd[0]), gold):
                pass1 += 1
            preds = [extract_pred(c) for c in sm]
            if any(is_correct(p, gold) for p in preds):
                passk += 1
            valid = [p for p in preds if p != ""]
            if valid:
                maj = Counter(valid).most_common(1)[0][0]
                if is_correct(maj, gold):
                    majk += 1
        print(f"  [{which}] {min(s+bs, n)}/{n}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "pass@1": round(pass1 / n, 4),
        f"pass@{args.k}": round(passk / n, 4),
        f"maj@{args.k}": round(majk / n, 4),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "sft", "dpo"])
    ap.add_argument("--test", default=str(PROJECT_ROOT / "data" / "test.jsonl"))
    ap.add_argument("--sft", default=str(PROJECT_ROOT / "checkpoints" / "sft"))
    ap.add_argument("--dpo", default=str(PROJECT_ROOT / "checkpoints" / "dpo"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "metrics.json"))
    ap.add_argument("--num-test", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.test)][: args.num_test]
    print(f"Evaluating on {len(rows)} test questions | device={device}")

    results = {}
    for which in args.models:
        print(f"\n=== {which.upper()} ===")
        results[which] = evaluate_model(which, rows, args.sft, args.dpo, tok, device, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)

    # pretty table
    print("\n" + "=" * 52)
    cols = ["pass@1", f"pass@{args.k}", f"maj@{args.k}"]
    header = f"{'model':<8}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in args.models:
        r = results[m]
        print(f"{m:<8}" + "".join(f"{r[c]*100:>11.1f}%" for c in cols))
    print("=" * 52)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
