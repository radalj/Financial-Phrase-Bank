"""
Shared configuration for Financial Transformer model.
All hyperparameters are defined here to ensure consistency across training, validation, and inference.
"""

# Model architecture hyperparameters
MODEL_CONFIG = {
    "embed_dim": 256,        # Embedding dimension
    "num_heads": 8,          # Number of attention heads
    "ff_dim": 512,           # Feed-forward hidden dimension
    "num_layers": 4,         # Number of transformer blocks
    "max_seq_len": 128,      # Maximum sequence length
    "num_classes": 3,        # Number of output classes (negative, neutral, positive)
    "dropout": 0.1,          # Dropout probability (matches FinancialTransformer default)
}

# Training hyperparameters
TRAIN_CONFIG = {
    "batch_size": 16,        # From original train_model.py
    "learning_rate": 2e-4,   # From original train_model.py
    "weight_decay": 0.0,     # Not specified in original, so 0
    "label_smoothing": 0.0,  # Not specified in original, so 0
    "num_epochs": 10,        # From original train_model.py
    "patience": 5,           # Early stopping patience
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "grad_clip_norm": 1.0,
}

# Data configuration
DATA_CONFIGS = ["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"]

# Model save path
MODEL_PATH = "financial_transformer.pth"
