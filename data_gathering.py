import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset


class FinancialDataset(Dataset):
    """
    Custom Dataset class for Financial Phrase Bank data.
    Tokenizes sentences and returns attention masks along with labels.
    """
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]['sentence']
        label = self.data[index]['label']
        
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }




def load_and_prepare_data(configs=["sentences_allagree", "sentences_50agree", "sentences_66agree", "sentences_75agree"], batch_size=16):
    """
    Load dataset and prepare train/validation dataloaders.
    
    Args:
        configs: List of dataset configurations to use
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary with tokenizer, train_loader, val_loader, and data splits
    """
    # Load and combine datasets from multiple configurations
    combined_data = []
    for config in configs:
        dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
        combined_data.extend(dataset['train'])

    # Split into train and validation (80-20)
    train_test_split = torch.utils.data.random_split(combined_data, [int(0.8 * len(combined_data)), len(combined_data) - int(0.8 * len(combined_data))])
    train_data = train_test_split[0]
    val_data = train_test_split[1]

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create DataLoaders
    train_loader = DataLoader(
        FinancialDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        FinancialDataset(val_data, tokenizer),
        batch_size=batch_size
    )

    print(f"Data loaded. Train size: {len(train_data)}, Val size: {len(val_data)}")

    return {
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_data': train_data,
        'val_data': val_data
    }


def load_test_data(configs=["sentences_allagree"], batch_size=16, tokenizer=None):
    """
    Load dataset and prepare train/validation/test dataloaders for final evaluation.
    
    Args:
        configs: List of dataset configurations to use
        batch_size: Batch size for DataLoader
        tokenizer: BertTokenizer instance
        
    Returns:
        Dictionary with tokenizer, train_loader, val_loader, test_loader, and data splits
    """
    # Load and combine datasets from multiple configurations
    combined_data = []
    for config in configs:
        dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
        combined_data.extend(dataset['train'])

    # 80% train, 10% val, 10% test
    train_size = int(0.8 * len(combined_data))
    val_test_size = len(combined_data) - train_size
    train_val = torch.utils.data.random_split(combined_data, [train_size, val_test_size])

    val_size = int(0.5 * val_test_size)
    val_test = torch.utils.data.random_split(train_val[1], [val_size, val_test_size - val_size])

    train_data = train_val[0]
    val_data = val_test[0]
    test_data = val_test[1]

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create DataLoaders
    train_loader = DataLoader(
        FinancialDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        FinancialDataset(val_data, tokenizer),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        FinancialDataset(test_data, tokenizer),
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Data loaded. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }


if __name__ == "__main__":
    
    # Load and prepare train/val data
    data = load_and_prepare_data(["sentences_allagree", "sentences_50agree", "sentences_66agree", "sentences_75agree"], batch_size=16)
    print(f"Tokenizer vocab size: {data['tokenizer'].vocab_size}")
