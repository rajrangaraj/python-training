"""
Demonstration of NLP tasks using transformers and BERT.
"""

import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    pipeline
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class NLPConfig:
    """Configuration for NLP tasks."""
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TextClassificationDataset(Dataset):
    """Custom dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def demonstrate_text_classification():
    """Demonstrate text classification with BERT."""
    
    # Sample data
    texts = [
        "I love this product, it's amazing!",
        "This is the worst purchase ever.",
        "The quality is good but expensive.",
        "Don't waste your money on this.",
        "Excellent service and fast delivery!"
    ]
    labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Create sentiment analysis pipeline
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    results = classifier(texts)
    
    return results

def demonstrate_named_entity_recognition():
    """Demonstrate Named Entity Recognition (NER) with BERT."""
    
    # Initialize NER pipeline
    ner = pipeline('ner', aggregation_strategy='simple')
    
    # Sample text
    text = """
    Apple Inc. CEO Tim Cook announced new products at their headquarters 
    in Cupertino, California. The event was also attended by Steve Wozniak.
    """
    
    # Perform NER
    entities = ner(text)
    
    return entities

def demonstrate_question_answering():
    """Demonstrate Question Answering with BERT."""
    
    # Initialize QA pipeline
    qa = pipeline('question-answering')
    
    # Sample context and questions
    context = """
    The Python programming language was created by Guido van Rossum 
    and was released in 1991. Python is known for its simplicity 
    and readability. It has become one of the most popular 
    programming languages for data science and machine learning.
    """
    
    questions = [
        "Who created Python?",
        "When was Python released?",
        "What is Python known for?"
    ]
    
    # Get answers
    answers = [
        qa(question=q, context=context)
        for q in questions
    ]
    
    return answers

def demonstrate_text_generation():
    """Demonstrate text generation with GPT-2."""
    
    # Initialize text generation pipeline
    generator = pipeline('text-generation', model='gpt2')
    
    # Sample prompts
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in Silicon Valley",
        "The best way to learn programming is"
    ]
    
    # Generate text
    generations = generator(
        prompts,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7
    )
    
    return generations

def visualize_attention_weights(text: str, model, tokenizer):
    """Visualize BERT attention weights."""
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention = outputs.attentions[-1][0]  # Last layer attention
    
    # Convert tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention[0].numpy(),  # First attention head
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis'
    )
    plt.title('BERT Attention Weights')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    return plt

if __name__ == '__main__':
    # Text Classification
    print("\nText Classification Results:")
    classification_results = demonstrate_text_classification()
    for text, result in zip(texts, classification_results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")
    
    # Named Entity Recognition
    print("\nNamed Entity Recognition Results:")
    ner_results = demonstrate_named_entity_recognition()
    for entity in ner_results:
        print(f"Entity: {entity['word']}")
        print(f"Type: {entity['entity_group']}")
        print(f"Score: {entity['score']:.4f}\n")
    
    # Question Answering
    print("\nQuestion Answering Results:")
    qa_results = demonstrate_question_answering()
    for q, a in zip(questions, qa_results):
        print(f"Question: {q}")
        print(f"Answer: {a['answer']}")
        print(f"Score: {a['score']:.4f}\n")
    
    # Text Generation
    print("\nText Generation Results:")
    generation_results = demonstrate_text_generation()
    for prompt, generation in zip(prompts, generation_results):
        print(f"Prompt: {prompt}")
        print(f"Generated: {generation[0]['generated_text']}\n")
    
    # Attention Visualization
    sample_text = "The transformer model revolutionized natural language processing."
    attention_plot = visualize_attention_weights(
        sample_text,
        BertForSequenceClassification.from_pretrained('bert-base-uncased'),
        BertTokenizer.from_pretrained('bert-base-uncased')
    )
    plt.show() 