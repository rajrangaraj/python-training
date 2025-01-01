"""
Demonstration of Natural Language Processing techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from collections import Counter
import re

@dataclass
class NLPConfig:
    """Configuration for NLP tasks."""
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42

class TextDataset(Dataset):
    """Custom dataset for text data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    """BERT-based text classifier."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(texts: List[str], nlp) -> Dict[str, List[Any]]:
    """Extract linguistic features using spaCy."""
    
    features = {
        'tokens': [],
        'pos_tags': [],
        'entities': [],
        'dependencies': []
    }
    
    for text in texts:
        doc = nlp(text)
        
        # Tokenization and POS tagging
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Dependency Parsing
        dependencies = [(token.text, token.dep_, token.head.text)
                       for token in doc]
        
        features['tokens'].append(tokens)
        features['pos_tags'].append(pos_tags)
        features['entities'].append(entities)
        features['dependencies'].append(dependencies)
    
    return features

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: NLPConfig
) -> Dict[str, List[float]]:
    """Train NLP model."""
    
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100. * train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(100. * val_correct / val_total)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{config.num_epochs}]")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history

def analyze_text(texts: List[str], nlp) -> Dict[str, Any]:
    """Analyze text characteristics."""
    
    analysis = {}
    
    # Token statistics
    all_tokens = []
    sentence_lengths = []
    
    for text in texts:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        all_tokens.extend(tokens)
        sentence_lengths.append(len(tokens))
    
    # Vocabulary analysis
    vocab = Counter(all_tokens)
    analysis['vocab_size'] = len(vocab)
    analysis['most_common'] = vocab.most_common(10)
    
    # Length statistics
    analysis['avg_length'] = np.mean(sentence_lengths)
    analysis['max_length'] = max(sentence_lengths)
    analysis['min_length'] = min(sentence_lengths)
    
    # Part of speech distribution
    pos_dist = Counter()
    for text in texts:
        doc = nlp(text)
        pos_dist.update([token.pos_ for token in doc])
    
    analysis['pos_distribution'] = dict(pos_dist)
    
    return analysis

def visualize_results(
    history: Dict[str, List[float]],
    analysis: Dict[str, Any]
):
    """Visualize NLP results."""
    
    plt.figure(figsize=(15, 5))
    
    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # POS distribution
    plt.subplot(1, 3, 2)
    pos_data = analysis['pos_distribution']
    plt.bar(pos_data.keys(), pos_data.values())
    plt.title('POS Distribution')
    plt.xticks(rotation=45)
    
    # Word frequency
    plt.subplot(1, 3, 3)
    words, counts = zip(*analysis['most_common'])
    plt.bar(words, counts)
    plt.title('Most Common Words')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = NLPConfig()
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Sample data (replace with your dataset)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating.",
        "Deep learning revolutionized NLP tasks."
    ]
    labels = [0, 1, 1]  # Binary classification example
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Extract features
    features = extract_features(processed_texts, nlp)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = TextDataset(processed_texts, labels, tokenizer, config.max_length)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create and train model
    model = BertClassifier(num_classes=2)
    history = train_model(model, loader, loader, config)  # Using same loader for demo
    
    # Analyze text
    analysis = analyze_text(processed_texts, nlp)
    
    # Visualize results
    visualization = visualize_results(history, analysis)
    visualization.show() 