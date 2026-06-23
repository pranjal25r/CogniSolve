"""
build_preferences.py — Generate DPO preference pairs from the SFT model itself
(rejection-sampling / self-improvement style).

For each training question we sample K completions from the SFT model. Using the
gold answer we label each completion correct/incorrect. A question yields a
preference pair only if it produced >=1 correct AND >=1 incorrect completion:
    chosen   = a correct completion
    rejected = an incorrect completion
This gives clean, on-policy preference data without any human labeling — the
model is taught to prefer its own correct reasoning over its own mistakes.

Output: data/prefs.jsonl  -> {"prompt", "chosen", "rejected"}

Run:
  python src/build_preferences.py --num-questions 1500 --k 4
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "distilgpt2"
# MUST match prepare_data.py
PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


def extract_pred(text: str) -> str:
    """Final numeric answer from a generated solution: prefer text after '####', else last number."""
    if "####" in text:
        text = text.split("####")[-1]
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def is_correct(pred: str, gold: str) -> bool:
    if pred == "" or gold == "":
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(PROJECT_ROOT / "checkpoints" / "sft"))
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "sft_train.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "prefs.jsonl"))
    ap.add_argument("--num-questions", type=int, default=1500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched generation

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, args.sft)
    model = model.merge_and_unload()  # bake SFT adapter into weights for fast generation
    model.config.pad_token_id = tok.eos_token_id
    model.to(device).eval()

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    rows = rows[: args.num_questions]

    pairs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        prompts = [r["prompt"] for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.eos_token_id,
            )
        # gen shape: [batch * k, seq]; strip the prompt portion
        prompt_len = enc["input_ids"].shape[1]
        completions = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)

        for i, r in enumerate(batch):
            cands = completions[i * args.k: (i + 1) * args.k]
            correct = [c for c in cands if is_correct(extract_pred(c), r["gold"])]
            wrong = [c for c in cands if not is_correct(extract_pred(c), r["gold"])]
            if correct and wrong:
                pairs.append({
                    "prompt": r["prompt"],
                    "chosen": " " + correct[0].strip(),
                    "rejected": " " + wrong[0].strip(),
                })

        if (start // args.batch_size) % 10 == 0:
            print(f"  processed {start + len(batch)}/{len(rows)} | pairs so far: {len(pairs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} preference pairs -> {out}")
    if len(pairs) < 100:
        print("WARNING: few pairs. Increase --num-questions or --k, or train SFT longer.")


if __name__ == "__main__":
    main()
