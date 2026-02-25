import torch
import torch.nn as nn
import torch.optim as optim
from data_gathering import load_and_prepare_data
from financial_transformer import FinancialTransformer


def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device='cpu', save_path="financial_transformer.pth"):
    """
    Train the Financial Transformer model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        device: Device to train on (cpu/cuda)
        save_path: Path to save trained model
    """
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully to {save_path}")


def main():
    """
    Main training script for Financial Transformer model.
    """
    # Load and prepare data
    data = load_and_prepare_data(configs=["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"], batch_size=16)
    train_loader = data['train_loader']
    tokenizer = data['tokenizer']

    # Model hyperparameters
    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    num_heads = 2
    ff_dim = 128
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

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00080346, weight_decay=4.877e-03)

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device,
        save_path="financial_transformer.pth"
    )


if __name__ == "__main__":
    main()
