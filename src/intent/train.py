"""Fine-tune DistilBERT on the JANET intent dataset and report how it did.

Run from the `src/` directory (matches JANET's import convention):

    cd src && python -m intent.train

What this does, end to end:
  1. Load the 13 labels and the train/val utterances (via dataset.py).
  2. Load a PRE-TRAINED DistilBERT and attach a fresh 13-way classifier head.
     We are fine-tuning: the model already understands English, we only teach
     it our categories, which is why ~2k examples is enough.
  3. Train for a few epochs on the GPU, using a class-weighted loss so the big
     NONE class doesn't drown out the smaller intents.
  4. Evaluate on the held-out val split: overall accuracy, per-class
     precision/recall/F1, and a confusion-matrix image so you can SEE exactly
     which intents get confused.
  5. Save the model + tokenizer to data/models/intent-distilbert/ so it can be
     loaded later (offline) to replace the keyword matcher in intent.py.

Nothing here is wired into the live assistant yet — this only produces and
measures the model.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from intent.dataset import IntentDataset, load_labels, read_split

# --- Configuration (kept together so every knob is visible in one place) -----
MODEL_NAME = "distilbert-base-uncased"   # the plan's choice; downloads once, then cached
EPOCHS = 4                               # DistilBERT fine-tunes fast; 4 passes is plenty here
BATCH_SIZE = 16
LEARNING_RATE = 2e-5                     # standard fine-tuning lr for BERT-family models
MAX_LENGTH = 32                          # spoken commands are short
SEED = 42                                # fixed so results are reproducible run-to-run

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "models" / "intent-distilbert"


class WeightedTrainer(Trainer):
    """A Trainer that uses a class-weighted cross-entropy loss.

    NONE has ~5x the examples of any single intent. Without weighting, the model
    could score well just by leaning toward NONE. Weighting each class by the
    inverse of its frequency tells the model to care about rare classes too.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device)
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Turn raw model outputs into readable scores for the Trainer's eval log."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),      # treats every class equally
        "f1_weighted": f1_score(labels, preds, average="weighted"),  # weights by class size
    }


def main():
    torch.manual_seed(SEED)

    # 1. Labels + the id<->name maps the model needs to store.
    labels = load_labels()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"{len(labels)} labels: {', '.join(labels)}")

    # 2. Tokenizer + data.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels = read_split("train", label2id)
    val_texts, val_labels = read_split("val", label2id)
    print(f"train: {len(train_texts)} examples | val: {len(val_texts)} examples")

    train_ds = IntentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = IntentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # 3. Class weights = total / (num_classes * count) — the "balanced" scheme.
    counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (len(labels) * counts[i]) for i in range(len(labels))],
        dtype=torch.float,
    )

    # 4. Pre-trained DistilBERT + a fresh 13-way head (randomly initialized).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        seed=SEED,
        fp16=torch.cuda.is_available(),   # half precision on GPU = faster, less memory
        report_to="none",                 # no external experiment trackers
        disable_tqdm=True,                # cleaner logs (per-epoch summaries, no step bars)
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # 5. Train.
    print("\n=== training ===")
    trainer.train()

    # 6. Evaluate in detail on the val split.
    print("\n=== validation results ===")
    pred_output = trainer.predict(val_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    print(f"accuracy: {accuracy_score(val_labels, preds):.3f}")
    print(f"macro F1: {f1_score(val_labels, preds, average='macro'):.3f}\n")
    print(classification_report(val_labels, preds, target_names=labels, digits=3, zero_division=0))

    # 7. Confusion matrix image — rows = true label, cols = predicted label.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(
        val_labels, preds, display_labels=labels, xticks_rotation="vertical", colorbar=False
    )
    disp.figure_.set_size_inches(11, 10)
    disp.figure_.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    disp.figure_.savefig(cm_path, dpi=120)
    print(f"\nconfusion matrix saved -> {cm_path}")

    # 8. Save the model + tokenizer for later use (loads offline).
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"model saved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
