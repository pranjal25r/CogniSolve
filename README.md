# CogniSolve

## What it is

CogniSolve teaches a tiny language model (DistilGPT-2, 82 million weights) to solve school
maths word problems. It trains the model in two stages and measures how much each stage
helps.

## Why I built it

To see how much two standard training methods, fine-tuning and DPO, help a very small
model.

## How it works

1. **Fine-tune on worked solutions.** The model is trained on step-by-step solutions from
   the GSM8K dataset. Only about 1% of the weights are trained, using LoRA.
2. **Collect the model's own right and wrong answers.** The fine-tuned model answers
   training questions 4 times each. Where it gets a question both right and wrong, one
   right answer and one wrong answer are kept as a pair. This gave 99 pairs.
3. **DPO.** For each pair, the model is nudged towards the right answer and away from the
   wrong one. The DPO loss is written directly in PyTorch.
4. **Test.** All three versions (original, fine-tuned, fine-tuned + DPO) are tested on 500
   questions they have not seen.

## Results

| Model | Pass@1 | Pass@4 | maj@4 |
|-------|:------:|:------:|:-----:|
| DistilGPT-2 (base) | 0.0% | 0.0% | 0.0% |
| + fine-tuning (LoRA) | 1.0% | 6.4% | 2.0% |
| + DPO              | 1.0% | 5.0% | 1.4% |

Tested on 500 GSM8K test questions.

- **Pass@1**: the model gives one answer, and it is right.
- **Pass@4**: the model gives four answers, and at least one is right.
- **maj@4**: the model gives four answers, and the most common one is right.

## What it means

- **Fine-tuning mainly taught the answer format.** An answer only counts if it comes after
  the `####` marker that GSM8K solutions use. The base model never writes that marker, so
  it scores 0 by design. After fine-tuning the model writes the marker and gets a few
  questions right.
- **DPO did not improve results.** On Pass@4 the fine-tuned model got 32 of 500 right and
  the DPO model got 25 of 500. That difference is within noise, so this is "no
  improvement", not "DPO made it worse". The likely reason is that 99 pairs is too few.

## Known limits

- **Tiny model.** An 82M model is far too small to be good at these problems. The scores
  only show how the stages compare with each other.
- **Small numbers.** With 500 questions and scores this low, a few questions either way
  changes the result.
- **The training pairs used a looser answer check than the final grading.** When picking
  pairs, an answer counted as right if the last number in the text matched, even with no
  `####` marker. The final grading only accepts the number straight after `####`. So a few
  "right" examples in the pairs may not count as right in grading.
- **Results file.** `results/metrics.json` is the output of the run reported above. The
  trained models are not in this repository, and the test answers are sampled without a
  fixed seed, so a rerun will give slightly different numbers.

## How to run

Takes about 1.5 to 2.5 hours on one T4 GPU (for example on Kaggle).

```bash
pip install -r requirements.txt

python src/prepare_data.py                       # → data/sft_train.jsonl, data/test.jsonl
python src/train_sft.py --epochs 3               # → checkpoints/sft
python src/build_preferences.py --num-questions 1500 --k 4   # → data/prefs.jsonl
python src/train_dpo.py --epochs 2 --beta 0.1    # → checkpoints/dpo
python src/evaluate.py --models base sft dpo --num-test 500 --k 4   # → results/metrics.json
```

Each script runs on its own and has command-line options for batch size, number of
questions and so on. On Kaggle, checkpoints are lost when the session ends unless you
save a version.

Built with PyTorch, Hugging Face `transformers`, `peft` and `datasets`.
