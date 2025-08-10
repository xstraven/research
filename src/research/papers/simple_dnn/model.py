import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(SimpleDNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Flatten input if needed (for image data)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.output_layer(x)
        return x


class TextClassificationDNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, num_classes, dropout_rate=0.2):
        super(TextClassificationDNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        self.layers = nn.ModuleList()
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            
        self.output_layer = nn.Linear(prev_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = self.embedding_dropout(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, embedding_dim)
        
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.output_layer(x)
        return x


def create_model(model_type, config):
    if model_type == "simple":
        return SimpleDNN(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            dropout_rate=config.dropout_rate
        )
    elif model_type == "text":
        return TextClassificationDNN(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")