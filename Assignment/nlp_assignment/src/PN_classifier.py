import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, tokenizer):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Resize the token embeddings to accommodate new special tokens
        self.bert.resize_token_embeddings(len(tokenizer))
        self.drop1 = nn.Dropout(p=0.3)  # First dropout layer
        self.fc1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2)  # Fully connected layer
        self.drop2 = nn.Dropout(p=0.2)  # Second dropout layer
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size // 2)  # Layer normalization
        self.out = nn.Linear(self.bert.config.hidden_size // 2, n_classes)  # Final output layer
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
            attention_mask (torch.Tensor): Tensor of attention masks to avoid performing attention on padding token indices.
            
        Returns:
            torch.Tensor: The logits for each input.
        """
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the pooled output of BERT
        
        # Pass through the first dropout layer
        dropped_output = self.drop1(pooled_output)
        
        # Pass through the first fully connected layer and apply ReLU activation
        fc1_output = F.relu(self.fc1(dropped_output))
        
        # Pass through the second dropout layer
        dropped_output = self.drop2(fc1_output)
        
        # Apply layer normalization
        norm_output = self.layer_norm(dropped_output)
        
        # Final output layer
        return self.out(norm_output)