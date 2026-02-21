import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from data_gathering import load_test_data
from financial_transformer import FinancialTransformer


def validate_model(model, data_loader, criterion, device, dataset_name="Validation"):
    """
    Validate/evaluate model on a given dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on (cpu/cuda)
        dataset_name: Name of dataset (for printing)
    
    Returns:
        Dictionary with loss, accuracy, predictions, and labels
    """
    model.eval()

    eval_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            eval_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_loss /= len(data_loader)
    eval_acc = correct / total

    print(f"\n{dataset_name} Results:")
    print(f"Loss: {eval_loss:.4f}")
    print(f"Accuracy: {eval_acc:.4f}")

    return {
        'loss': eval_loss,
        'accuracy': eval_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def calculate_metrics(labels, predictions):
    """
    Calculate precision, recall, and F1-score.
    
    Args:
        labels: True labels
        predictions: Model predictions
    
    Returns:
        Dictionary with calculated metrics
    """
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(labels, predictions, title="Confusion Matrix", filename=None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        labels: True labels
        predictions: Model predictions
        title: Title for the plot
        filename: Filename to save plot (optional)
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def main():
    """
    Main validation and testing script.
    Load trained model and evaluate on validation and test sets.
    """
    # Load and prepare data
    data = load_test_data(config="sentences_allagree", batch_size=16)
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    tokenizer = data['tokenizer']

    # Model hyperparameters
    vocab_size = tokenizer.vocab_size
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 4
    max_seq_len = 128
    num_classes = 3

    # Initialize model
    model = FinancialTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_classes=num_classes
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load trained model
    model_path = "financial_transformer.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from {model_path}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ===== Validation Evaluation =====
    print("\n" + "="*50)
    print("VALIDATION EVALUATION")
    print("="*50)
    
    val_results = validate_model(model, val_loader, criterion, device, "Validation")
    print("\nValidation Metrics:")
    val_metrics = calculate_metrics(val_results['labels'], val_results['predictions'])
    plot_confusion_matrix(
        val_results['labels'],
        val_results['predictions'],
        title="Confusion Matrix - Validation Set",
        filename="confusion_matrix_validation.png"
    )

    # ===== Test Evaluation =====
    print("\n" + "="*50)
    print("TEST EVALUATION")
    print("="*50)
    
    test_results = validate_model(model, test_loader, criterion, device, "Test")
    print("\nTest Metrics:")
    test_metrics = calculate_metrics(test_results['labels'], test_results['predictions'])
    plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        title="Confusion Matrix - Test Set",
        filename="confusion_matrix_test.png"
    )

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nValidation Performance:")
    print(f"  Loss: {val_results['loss']:.4f}")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  F1-score: {val_metrics['f1']:.4f}")
    
    print("\nTest Performance:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  F1-score: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
