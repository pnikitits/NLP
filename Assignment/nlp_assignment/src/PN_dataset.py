import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    Dataset class for sentiment analysis with highlighted target terms.
    """
    def __init__(self, modified_sentences, labels, tokenizer, max_len=128):
        self.sentences = modified_sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        modified_sentence = str(self.sentences[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            modified_sentence,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }