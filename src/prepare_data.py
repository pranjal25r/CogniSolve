"""
prepare_data.py — Download GSM8K and format it for chain-of-thought SFT.

GSM8K answers already contain step-by-step reasoning ending in "#### <number>".
We keep that format so the model learns to (a) reason step by step and
(b) emit a parseable final answer after "####".

Outputs (under data/):
  - sft_train.jsonl   : {"prompt": ..., "answer": ..., "gold": ...}
  - test.jsonl        : same schema, held-out test split

Run:
  python src/prepare_data.py
  python src/prepare_data.py --max-train 5000   # optional cap for faster runs
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Prompt format. MUST stay identical across prepare_data / build_preferences / evaluate. ---
PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


def extract_gold(answer_text: str) -> str:
    """Pull the final numeric answer that follows '####' in a GSM8K answer."""
    if "####" in answer_text:
        tail = answer_text.split("####")[-1]
    else:
        tail = answer_text
    tail = tail.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+\.?\d*", tail)
    return m[-1] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=None,
                    help="Optional cap on number of training examples.")
    ap.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "data"))
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GSM8K (openai/gsm8k, config 'main') ...")
    ds = load_dataset("openai/gsm8k", "main")

    def write_split(split_name, rows, cap=None):
        path = out_dir / f"{split_name}.jsonl"
        n = 0
        with open(path, "w") as f:
            for ex in rows:
                if cap is not None and n >= cap:
                    break
                q = ex["question"].strip()
                a = ex["answer"].strip()
                rec = {
                    "prompt": PROMPT_TEMPLATE.format(q=q),
                    "answer": " " + a,          # leading space => clean tokenization after "Answer:"
                    "gold": extract_gold(a),
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
        print(f"  wrote {n} -> {path}")

    write_split("sft_train", ds["train"], cap=args.max_train)
    write_split("test", ds["test"], cap=None)
    print("Done.")


if __name__ == "__main__":
    main()
