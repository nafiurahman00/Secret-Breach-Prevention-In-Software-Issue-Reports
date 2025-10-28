"""
TextCNN Ensemble Model for Secret Detection
Two separate TextCNN networks:
1. Candidate String Network - trains on candidate_string (16 epochs)
2. Context Window Network - trains on context_window (32 epochs)

Both use character-level input with one-hot encoding.
Final prediction is ensemble of both networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle


# Define character set - printable ASCII + special characters
def get_character_set():
    """Get the full character set for encoding"""
    # Printable ASCII characters
    chars = []
    for i in range(256):
        chars.append(chr(i))
    return sorted(list(set(chars)))


CHAR_SET = get_character_set()
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
VOCAB_SIZE = len(CHAR_SET)
print(f"Vocabulary size: {VOCAB_SIZE}")


class CharDataset(Dataset):
    """Dataset for character-level text classification"""
    
    def __init__(self, texts, labels=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Convert text to character indices
        char_indices = []
        for char in text[:self.max_len]:
            if char in CHAR_TO_IDX:
                char_indices.append(CHAR_TO_IDX[char])
            else:
                # Unknown character - use index 0
                char_indices.append(0)
        
        # Pad or truncate
        if len(char_indices) < self.max_len:
            char_indices += [0] * (self.max_len - len(char_indices))
        else:
            char_indices = char_indices[:self.max_len]
        
        char_tensor = torch.tensor(char_indices, dtype=torch.long)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return char_tensor, label
        else:
            return char_tensor


class TextCNN(nn.Module):
    """
    TextCNN model with 6 convolutional layers and 3 fully connected layers.
    Character-level input with one-hot encoding.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_classes=2):
        super(TextCNN, self).__init__()
        
        # Embedding layer (learnable, acts like one-hot encoding with dimension reduction)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 6 Convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=4, padding=2)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 3 Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1)
        x = x.squeeze(2)  # (batch_size, 128)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """Train the model"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, "
              f"Accuracy: {100.*correct/total:.2f}%")


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and probabilities"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            if isinstance(data, tuple):
                data = data[0]
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")
    
    # Prepare data for candidate_string network
    print("\n" + "="*50)
    print("Training Candidate String Network")
    print("="*50)
    
    candidate_train_dataset = CharDataset(
        train_df['candidate_string'].values,
        train_df['label'].values,
        max_len=256
    )
    candidate_test_dataset = CharDataset(
        test_df['candidate_string'].values,
        max_len=256
    )
    
    candidate_train_loader = DataLoader(
        candidate_train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )
    candidate_test_loader = DataLoader(
        candidate_test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )
    
    # Train candidate string network
    candidate_model = TextCNN(VOCAB_SIZE).to(device)
    candidate_criterion = nn.CrossEntropyLoss()
    candidate_optimizer = torch.optim.Adam(candidate_model.parameters(), lr=0.0001)
    
    train_model(
        candidate_model,
        candidate_train_loader,
        candidate_criterion,
        candidate_optimizer,
        device,
        epochs=16
    )
    
    # Save candidate model
    torch.save(candidate_model.state_dict(), 'candidate_textcnn_model.pth')
    print("Candidate model saved to 'candidate_textcnn_model.pth'")
    
    # Get predictions from candidate model
    print("\nGetting predictions from candidate model...")
    candidate_preds, candidate_probs = evaluate_model(
        candidate_model,
        candidate_test_loader,
        device
    )
    
    # Prepare data for context_window network
    print("\n" + "="*50)
    print("Training Context Window Network")
    print("="*50)
    
    context_train_dataset = CharDataset(
        train_df['context_window'].values,
        train_df['label'].values,
        max_len=512
    )
    context_test_dataset = CharDataset(
        test_df['context_window'].values,
        max_len=512
    )
    
    context_train_loader = DataLoader(
        context_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    context_test_loader = DataLoader(
        context_test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Train context window network
    context_model = TextCNN(VOCAB_SIZE).to(device)
    context_criterion = nn.CrossEntropyLoss()
    context_optimizer = torch.optim.Adam(context_model.parameters(), lr=0.0001)
    
    train_model(
        context_model,
        context_train_loader,
        context_criterion,
        context_optimizer,
        device,
        epochs=32
    )
    
    # Save context model
    torch.save(context_model.state_dict(), 'context_textcnn_model.pth')
    print("Context model saved to 'context_textcnn_model.pth'")
    
    # Get predictions from context model
    print("\nGetting predictions from context model...")
    context_preds, context_probs = evaluate_model(
        context_model,
        context_test_loader,
        device
    )
    
    # Ensemble predictions (average probabilities)
    print("\n" + "="*50)
    print("Ensemble Predictions")
    print("="*50)
    
    # Average the probabilities from both models
    ensemble_probs = (candidate_probs + context_probs) / 2
    ensemble_preds = ensemble_probs.argmax(axis=1)
    
    # Get true labels
    true_labels = test_df['label'].values
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT - ENSEMBLE MODEL")
    print("="*50)
    print(classification_report(true_labels, ensemble_preds, 
                                target_names=['Non-Secret (0)', 'Secret (1)'],
                                digits=4))
    
    # Also print individual model reports
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT - CANDIDATE STRING MODEL")
    print("="*50)
    print(classification_report(true_labels, candidate_preds,
                                target_names=['Non-Secret (0)', 'Secret (1)'],
                                digits=4))
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT - CONTEXT WINDOW MODEL")
    print("="*50)
    print(classification_report(true_labels, context_preds,
                                target_names=['Non-Secret (0)', 'Secret (1)'],
                                digits=4))
    
    # Save predictions
    results_df = test_df.copy()
    results_df['candidate_pred'] = candidate_preds
    results_df['context_pred'] = context_preds
    results_df['ensemble_pred'] = ensemble_preds
    results_df['candidate_prob_1'] = candidate_probs[:, 1]
    results_df['context_prob_1'] = context_probs[:, 1]
    results_df['ensemble_prob_1'] = ensemble_probs[:, 1]
    results_df.to_csv('textcnn_ensemble_predictions.csv', index=False)
    print("\nPredictions saved to 'textcnn_ensemble_predictions.csv'")


if __name__ == '__main__':
    main()
