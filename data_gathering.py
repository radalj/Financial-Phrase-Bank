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




def load_and_prepare_data(config="sentences_allagree", batch_size=16):
    """
    Load dataset and prepare train/validation dataloaders.
    
    Args:
        config: Dataset configuration to use
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary with tokenizer, train_loader, val_loader, and data splits
    """
    # Load dataset
    dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
    
    # Split into train and validation (80-20)
    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    train_data = train_test_split['train']
    val_data = train_test_split['test']

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


def load_test_data(config="sentences_allagree", batch_size=16, tokenizer=None):
    """
    Load dataset and prepare train/validation/test dataloaders for final evaluation.
    
    Args:
        config: Dataset configuration to use
        batch_size: Batch size for DataLoader
        tokenizer: BertTokenizer instance
        
    Returns:
        Dictionary with tokenizer, train_loader, val_loader, test_loader, and data splits
    """
    # Load dataset
    dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
    
    # 80% train, 10% val, 10% test
    train_val = dataset["train"].train_test_split(test_size=0.2)
    val_test = train_val["test"].train_test_split(test_size=0.5)

    train_data = train_val["train"]
    val_data = val_test["train"]
    test_data = val_test["test"]

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
    data = load_and_prepare_data("sentences_allagree", batch_size=16)
    print(f"Tokenizer vocab size: {data['tokenizer'].vocab_size}")
