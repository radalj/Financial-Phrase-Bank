import torch
import csv
import sys
import os
from transformers import BertTokenizer
from financial_transformer import FinancialTransformer
from config import MODEL_CONFIG, MODEL_PATH

# Label mapping: matches financial_phrasebank dataset (0=negative, 1=neutral, 2=positive)
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def predict(sentences_csv, model_path=MODEL_PATH, output_csv="tests/submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize model with config hyperparameters (automatically matches training)
    model = FinancialTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        ff_dim=MODEL_CONFIG["ff_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        num_classes=MODEL_CONFIG["num_classes"],
        dropout=MODEL_CONFIG["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Read sentences
    rows = []
    with open(sentences_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Running predictions on {len(rows)} sentences...")

    results = []
    with torch.no_grad():
        for row in rows:
            row_id = row["row_id"]
            sentence = row["sentence"]

            encoding = tokenizer(
                sentence,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            pred_idx = logits.argmax(dim=1).item()
            label = LABEL_MAP[pred_idx]
            results.append({"row_id": row_id, "label": label})

    # Write submission file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "label"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Submission saved to {output_csv} ({len(results)} rows)")

    # Print label distribution
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    sentences_csv = sys.argv[1] if len(sys.argv) > 1 else "tests/sentences.csv"
    predict(sentences_csv)
