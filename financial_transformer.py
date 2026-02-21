import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from datasets import load_dataset

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism from scratch.
    Computes attention scores between all positions in the sequence.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads for Multi-Head Attention
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            Tuple of:
              - output tensor of shape (batch_size, seq_len, embed_dim)
              - attention_weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V matrices
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores: QK^T / sqrt(d_k)
        # Using torch.matmul for matrix multiplication
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask to (batch_size, 1, 1, seq_len) to broadcast with scores (batch_size, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to V: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back: (batch_size, seq_len, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class LayerNorm(nn.Module):
    """
    Layer Normalization from scratch.
    Normalizes inputs across the feature dimension for each sample independently.
    Formula: y = (x - mean) / sqrt(variance + eps) * gamma + beta
    """
    def __init__(self, embed_dim, eps=1e-6):
        """
        Args:
            embed_dim: Dimension of input embeddings
            eps: Small constant for numerical stability
        """
        super(LayerNorm, self).__init__()
        self.eps = eps
        # Learnable parameters: gamma (scale) and beta (shift)
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        
    def forward(self, x):
        """
        Forward pass for layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Normalized tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Calculate mean across the last dimension (embed_dim)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate variance across the last dimension
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch_size, seq_len, 1)
        
        # Normalize: subtract mean and divide by standard deviation
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        
        # Apply learnable scale (gamma) and shift (beta)
        output = self.gamma * x_normalized + self.beta
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network from scratch.
    Applies two linear transformations with ReLU activation in between.
    Formula: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of input embeddings
            ff_dim: Dimension of hidden layer in feed-forward network
            dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        # First linear transformation: embed_dim -> ff_dim
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        # Second linear transformation: ff_dim -> embed_dim
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # First linear transformation
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, ff_dim)
        x = self.linear1(x)
        
        # Apply ReLU activation: max(0, x)
        x = F.relu(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Second linear transformation
        # Shape: (batch_size, seq_len, ff_dim) -> (batch_size, seq_len, embed_dim)
        x = self.linear2(x)
        
        # Apply dropout again
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block containing:
    - Multi-Head Self-Attention
    - Layer Normalization
    - Feed-Forward Network
    - Residual Connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of hidden layer in feed-forward network
            dropout: Dropout probability
        """
        super(TransformerBlock, self).__init__()
        
        # Self-Attention layer
        self.attention = SelfAttention(embed_dim, num_heads)
        
        # Custom Layer Normalization layers
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            Tuple of:
              - output tensor of shape (batch_size, seq_len, embed_dim)
              - attention_weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Self-Attention with residual connection and layer normalization
        attention_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))  # Residual connection
        
        # Feed-Forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Residual connection
        
        return x, attention_weights


class FinancialTransformer(nn.Module):
    """
    Complete Transformer model for financial text classification.
    Includes:
    - Token embeddings
    - Positional embeddings
    - Multiple Transformer blocks
    - Final classifier for 3 classes
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, 
                 num_layers, max_seq_len, num_classes=3, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward hidden layer
            num_layers: Number of transformer blocks
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes (default: 3)
            dropout: Dropout probability
        """
        super(FinancialTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding layer
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier layer for 3 classes
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass through the entire model.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional mask tensor
            return_attention: If True, also return attention weights from the last layer
            
        Returns:
            logits of shape (batch_size, num_classes), and optionally
            attention_weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = x.shape
        
        # Generate position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Get token embeddings and positional embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_len, embed_dim)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through all transformer blocks, keeping last layer's attention weights
        last_attention_weights = None
        for transformer_block in self.transformer_blocks:
            x, last_attention_weights = transformer_block(x, mask)
        
        # Mean pooling across sequence
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Final classification layer
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        if return_attention:
            return logits, last_attention_weights
        return logits
    



# Example usage
if __name__ == "__main__":

    # Model hyperparameters
    vocab_size = 10000      # Size of vocabulary
    embed_dim = 256         # Embedding dimension
    num_heads = 8           # Number of attention heads
    ff_dim = 1024           # Feed-forward dimension
    num_layers = 4          # Number of transformer blocks
    max_seq_len = 128       # Maximum sequence length
    num_classes = 3         # Number of output classes (Positive, Negative, Neutral)
    dropout = 0.1           # Dropout rate
    
    # Create model
    model = FinancialTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Example input: batch of token indices
    batch_size = 16
    seq_len = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
