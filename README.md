# MCQ Answer Ranking with Deep Learning

Predicting the top 3 most likely correct answers for multiple-choice science and philosophy questions, scored with **MAP@3**. Built for a Deep Learning & Generative AI course project and submitted to Kaggle.

| | |
|---|---|
| **Task** | Rank the top 3 of 5 options (A–E) for each question |
| **Metric** | MAP@3 (target: ≥ 0.73) |
| **Best model** | Bidirectional GRU + Multi-Head Attention |
| **Final Kaggle score** | **0.759**, ranked **102 / 1,356** teams |
| **Experiment tracking** | Weights & Biases |

---

## Table of Contents

1. Overview
2. Dataset
3. Preprocessing
4. Models
5. Results
6. Known Limitations
7. Future Work
8. Getting Started
9. Tech Stack
10. References
11. Full Report

---

## Overview

Given a question and five answer options, the model predicts which three options are most likely correct, in ranked order. **MAP@3** rewards a correct answer more the higher it is ranked: 1.0 for rank 1, 0.5 for rank 2, 0.333 for rank 3, and 0 otherwise, averaged over all questions.

Four different approaches were built and compared:

1. TF-IDF + feed-forward neural network (baseline)
2. Fine-tuned DistilBERT (per-option binary classifier)
3. Weighted ensemble of Models 1 and 2
4. Word embeddings + BiGRU with multi-head attention (**best**)

## Dataset

- **Train:** 2,000 labeled questions
- **Test:** 500 unlabeled questions
- Each question has 5 options (A–E); no missing values.
- Topics: physics, astrophysics, chemistry, mathematics, and philosophy.

**Key EDA findings**

- **Mild label imbalance:** B (24.5%) and C (23.0%) are the most common correct answers; E (16.2%) is the least.
- **Templated questions:** many start with openers like "Pick the best possible answer:", which were stripped before modeling.
- **Similar options:** 54% of questions (1,078 / 2,000) have options with very similar wording, so plain keyword matching is unreliable. This motivated attention-based models.
- **Duplicates:** 344 exact duplicate rows were intentionally kept in training.
- **Length:** prompts average ~118 characters and options ~162–167 characters. Correct-answer length shows no consistent pattern, so length is not a usable shortcut.

## Preprocessing

Applied consistently across all four models:

1. Strip instructional prefixes from questions.
2. Combine each question with its options into a single text field.
3. 80/20 train/validation split with `GroupShuffleSplit` (grouped by question): 1,576 train / 424 validation rows with no question overlap.
4. Encode labels A–E as 0–4.

**Model-specific input**

- **Model 1:** `TfidfVectorizer` (5,000 features, unigrams + bigrams, `min_df=2`).
- **Model 2:** DistilBERT's built-in subword tokenizer.
- **Model 4:** Keras word-level `Tokenizer` (10,000-word vocabulary). Each row is formatted as `question [SEP] A [SEP] B [SEP] C [SEP] D [SEP] E` and padded/truncated to 300 tokens. The fitted tokenizer is saved with `pickle` for inference.

## Models

### Model 1: TF-IDF + Feed-Forward NN (baseline)

`Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax)`, trained with Adam and sparse categorical cross-entropy. Fast to train and easy to audit. Its perfect validation accuracy is flagged as likely data leakage (see [Known Limitations](#known-limitations)).

### Model 2: Fine-tuned DistilBERT

Each `(question, option)` pair is scored independently as correct/incorrect using `AutoModelForSequenceClassification` (`num_labels=2`). The five scores are then sorted to produce the top-3 ranking.

- Optimizer: AdamW, learning rate 2e-5 (no scheduler)
- Batch size 16, 40 epochs
- First 4 of 6 transformer layers frozen; last 2 fine-tuned
- No quantization applied (dynamic INT8 quantization is a possible next step)

### Model 3: Ensemble (Model 1 + Model 2)

Blended probabilities: `0.3 × Model 1 + 0.7 × Model 2`. The weights were hand-picked rather than tuned on validation data, and the ensemble did **not** beat Model 1 alone.

### Model 4: Embeddings + BiGRU + Attention (best)

```
Embedding → SpatialDropout1D(0.2)
  → BiGRU(128, return sequences)
  → Multi-Head Attention (4 heads) + residual connection
  → BiGRU(64)
  → Dense(64, ReLU) → Dropout(0.3)
  → Dense(32, ReLU) → Dropout(0.2)
  → Dense(5, Softmax)
```

- Adam (lr 1e-3), up to 500 epochs
- Early stopping (patience 11, restore best weights)
- Reduce-LR-on-plateau (halve the rate after 8 stalled epochs)
- Balanced class weights via `compute_class_weight` to counter the B/C-heavy label distribution
- Pretrained FastText embeddings were intended, but the download failed, so the embedding layer used **random initialization**

## Results

| Model | Architecture | Train Acc | Val Acc | Val Loss | MAP@3 |
|---|---|---|---|---|---|
| 1 | TF-IDF + ANN | 1.000* | 1.000* | 3.7e-06* | 0.73* |
| 2 | Fine-tuned DistilBERT | n/a | n/a | n/a | 0.74 |
| 3 | Ensemble (1 + 2) | n/a | n/a | n/a | 0.73 |
| 4 | **BiGRU + Attention** | 0.956 | 0.960 | 0.079 | **0.759** |

\* Likely inflated by data leakage; do not treat as a reliable estimate.

**Why Model 4 is the most trustworthy result:** its accuracy climbed gradually over training, train and validation tracked closely, and its MAP@3 was consistent across repeated runs. Its final Kaggle score of 0.759 matches the validation range, whereas Model 1's inflated figure does not.

## Known Limitations

- **Probable data leakage in Model 1.** Validation accuracy hit 1.000 by epoch 3 and stayed flat, most likely from near-duplicate questions shared across splits. Its MAP@3 should be treated as optimistic until checked at the token level.
- **No precision/recall/F1.** Only MAP@3, accuracy and loss were tracked, so per-class weaknesses (e.g. on D and E) are not visible.
- **No pretrained embeddings in Model 4.** The transfer-learning benefit was never actually tested.
- **Single train/validation split.** No cross-validation, so scores carry some uncertainty.
- **Model 2 logged training loss only**, with no per-epoch validation curve.
- **Class weighting only in Model 4**, so Models 1–3 are not directly comparable on that point.

## Future Work

- Investigate and fix the likely leakage behind Model 1's perfect accuracy.
- Add per-class precision, recall and F1.
- Load pretrained FastText embeddings correctly.
- Tune the ensemble weights against validation MAP@3.
- Use grouped k-fold cross-validation.
- Try RoBERTa / DeBERTa with a fuller fine-tuning budget.

## Getting Started

> Fill in the paths below to match your repository.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Install dependencies
pip install torch tensorflow transformers datasets scikit-learn \
            pandas numpy matplotlib seaborn wandb gensim

# 3. Add the competition data (train.csv, test.csv) to the data directory

# 4. Run the notebook(s)
jupyter notebook <your-notebook>.ipynb
```

The project was trained in a Kaggle notebook environment; library versions followed those available there. Exact pinned versions can be taken from the original notebook's environment output. All runs (architecture, hyperparameters, validation MAP@3) are logged in Weights & Biases.

## Tech Stack

PyTorch, TensorFlow/Keras, Hugging Face Transformers & Datasets, Sentence-Transformers, FAISS, Gensim, scikit-learn, Weights & Biases, pandas, NumPy, Matplotlib, Seaborn.

## References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* arXiv:1406.1078.
- Hugging Face model card: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

## Full Report

See `Project_Report_submission.docx` for the complete write-up, including EDA charts, architecture diagrams, and Weights & Biases training curves.
