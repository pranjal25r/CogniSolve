# CogniSolve — SFT + DPO for Mathematical Reasoning

Teaching a small language model (**DistilGPT-2, 82M**) to reason through grade-school
math word problems, then aligning it to prefer its *correct* chains of thought over its
*incorrect* ones — using **LoRA fine-tuning** and a **from-scratch implementation of
Direct Preference Optimization (DPO)**.

This project is about **methodology and measured lift**, not leaderboard numbers: an 82M
model is far too small to "solve" GSM8K. The point is a clean, reproducible
SFT → preference-data → DPO → evaluation pipeline, and an honest measurement of how much
each stage moves Pass@K.

---

## Pipeline

```
GSM8K (CoT)
   │
   ▼
[1] SFT (LoRA)            distilgpt2  +  LoRA adapter   → learns CoT format + answer extraction
   │
   ▼
[2] Preference mining     sample K solutions per question from the SFT model;
   │                      label by final-answer match → (chosen=correct, rejected=wrong) pairs
   ▼
[3] DPO (from scratch)    fresh LoRA adapter on the SFT model; reference = SFT (adapter off)
   │                      loss = -logσ( β·[ (πc−πr) − (refc−refr) ] )
   ▼
[4] Evaluation            Pass@1 (greedy) · Pass@K (any-correct) · maj@K (self-consistency)
                          measured for base vs SFT vs DPO
```

**Why these choices**
- **LoRA/PEFT** — only ~0.5–1% of params are trainable; fast on a single T4 and the
  standard way fine-tuning is done in production.
- **Rejection-sampling DPO** — preference pairs are generated *on-policy* from the model's
  own samples, labeled automatically by the gold answer. No human annotation, and the model
  is taught to prefer its own correct reasoning over its own mistakes.
- **DPO from scratch** — the loss and the reference-policy log-probs are implemented
  directly in PyTorch. The SFT adapter is merged into the base, a fresh adapter becomes the
  trainable policy, and the reference forward pass simply *disables* that adapter — so no
  second model copy sits in memory.

---

## Results


| Model | Pass@1 | Pass@4 | maj@4 |
|-------|:------:|:------:|:-----:|
| DistilGPT-2 (base) | 0.0% | 0.0% | 0.0% |
| + SFT (LoRA)       | 1.0% | 6.4% | 2.0% |
| + DPO              | 1.0% | 5.0% | 1.4% |

*n = 500 GSM8K test questions. **Pass@1** = greedy decode; **Pass@K** = at least one of K
sampled solutions is correct; **maj@K** = majority-voted final answer across K samples.*

**Finding — SFT:** The base model scores 0% because it never emits the `####` answer
marker the evaluator requires. SFT teaches the chain-of-thought and answer format,
reaching 1.0% Pass@1 and 6.4% Pass@4.

**Finding — DPO:** With only 99 automatically-mined preference pairs, DPO did not improve
over SFT. The Pass@4 difference (6.4% vs 5.0%, i.e. 32 vs 25 of 500 questions) is within
noise at this sample size, so the result is "no improvement", not "DPO hurt". An 82M
model near ~6% Pass@4 produces too few correct solutions to mine enough preference pairs
for DPO to help.

---

## Reproduce (Kaggle T4, ~1.5–2.5 h end to end)

```bash
pip install -r requirements.txt

python src/prepare_data.py                       # → data/sft_train.jsonl, data/test.jsonl
python src/train_sft.py --epochs 3               # → checkpoints/sft
python src/build_preferences.py --num-questions 1500 --k 4   # → data/prefs.jsonl
python src/train_dpo.py --epochs 2 --beta 0.1    # → checkpoints/dpo
python src/evaluate.py --models base sft dpo --num-test 500 --k 4   # → results/metrics.json
```

All scripts are self-contained (inline helpers, no cross-module imports) and take CLI
overrides for batch size, dataset caps, K, etc. Checkpoints on Kaggle are ephemeral unless
you **Save Version** — commit one before the session recycles.

---

## Stack
PyTorch · Hugging Face `transformers` · `peft` (LoRA) · `datasets` · GSM8K · single T4 GPU

## Honest limitations
- 82M params caps absolute accuracy; this measures *relative* gains from each training stage,
  not competitive math performance.
- Preference data quality depends on the SFT model producing *some* correct samples; if it
  produces too few, increase `--num-questions` / `--k` or train SFT longer.
- DPO is run for a small number of epochs on automatically-labeled pairs; the goal is a
  measurable Pass@1 improvement, demonstrated honestly, not a maximized score.
