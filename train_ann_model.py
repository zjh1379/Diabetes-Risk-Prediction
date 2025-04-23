#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an Artificial Neural Network (ANN) model using the best features
selected from feature selection process for diabetes prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Project directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
REPORTS_DIR = os.path.join(CURRENT_DIR, 'ann_reports')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Custom Dataset for PyTorch
class DiabetesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural Network Model
class DiabetesANN(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, num_classes=3):
        super(DiabetesANN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_data(file_path, selected_features_path):
    """
    Load and preprocess the diabetes dataset
    
    Args:
        file_path: Path to the dataset CSV file
        selected_features_path: Path to the CSV file containing selected features
        
    Returns:
        X: Features data
        y: Target variable
        feature_names: Names of the selected features
    """
    print(f"Loading data from {file_path}")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Load selected features
    selected_features_df = pd.read_csv(selected_features_path)
    selected_features = selected_features_df['Feature'].tolist()
    
    # Get features and target
    X = df[selected_features]
    y = df['Diabetes_012']
    
    print(f"Data loaded with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Selected features: {selected_features}")
    
    return X, y, selected_features

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=50, patience=10):
    """
    Train the neural network model
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: Neural network model
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run the model on (CPU or GPU)
        num_epochs: Maximum number of epochs to train
        patience: Early stopping patience
        
    Returns:
        model: Trained model
        history: Training history (losses and metrics)
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # Track loss and accuracy
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_ann_model.pt'))
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, history

def evaluate_model(model, test_loader, criterion, device, class_names=None):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run the model on (CPU or GPU)
        class_names: Names of the target classes
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Track results
            test_loss += loss.item() * features.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # For multiclass, we use weighted F1 score
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # Print evaluation results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_targets)))]
    
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    print(report)
    
    # Create results dictionary
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'targets': all_targets,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }
    
    return results

def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Names of the target classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f"{conf_matrix[i, j]}\n({cm_norm[i, j]:.2f})",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()

def save_model_info(model, input_size, feature_names, class_names, results, save_path):
    """
    Save model information and results
    
    Args:
        model: Trained neural network model
        input_size: Number of input features
        feature_names: Names of the input features
        class_names: Names of the target classes
        results: Evaluation results
        save_path: Path to save the information
    """
    info = {
        'model_type': 'ANN (Artificial Neural Network)',
        'input_size': input_size,
        'feature_names': feature_names,
        'class_names': class_names,
        'model_structure': str(model),
        'accuracy': results['accuracy'],
        'f1_score': results['f1_score'],
        'classification_report': results['classification_report']
    }
    
    with open(save_path, 'w') as f:
        f.write("Diabetes Prediction Model Information\n")
        f.write("====================================\n\n")
        
        for key, value in info.items():
            f.write(f"{key}: {value}\n\n")
    
    print(f"Model information saved to {save_path}")

def main():
    print("Starting ANN model training for diabetes prediction...")
    
    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join(DATA_DIR, 'diabetes_012_health_indicators_BRFSS2015.csv', 
                           'diabetes_012_health_indicators_BRFSS2015.csv')
    best_features_path = os.path.join(CURRENT_DIR, 'feature_selection_output', 'best_features.csv')
    
    # If best_features.csv doesn't exist, try the reports directory
    if not os.path.exists(best_features_path):
        best_features_path = os.path.join(CURRENT_DIR, 'reports', 'best_features.csv')
    
    # If still doesn't exist, use a hard-coded feature list
    if not os.path.exists(best_features_path):
        print("Warning: best_features.csv not found. Using default feature list.")
        
        # These are the consensus features from the feature selection process
        selected_features = ['HighBP', 'HighChol', 'GenHlth', 'CholCheck', 'Age', 
                             'DiffWalk', 'HvyAlcoholConsump', 'HeartDiseaseorAttack', 
                             'BMI', 'Fruits', 'NoDocbcCost', 'Sex', 'Income']
        
        # Create a DataFrame to simulate loading from file
        selected_features_df = pd.DataFrame({'Feature': selected_features})
        
        # Save to be used by the model
        os.makedirs(os.path.dirname(best_features_path), exist_ok=True)
        selected_features_df.to_csv(best_features_path, index=False)
    
    # Load the dataset and selected features
    df = pd.read_csv(data_path)
    selected_features_df = pd.read_csv(best_features_path)
    selected_features = selected_features_df['Feature'].tolist()
    
    # Get features and target
    X = df[selected_features].values
    y = df['Diabetes_012'].values
    
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Selected features: {selected_features}")
    
    # Split into train, validation, test sets (70%, 15%, 15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for future use
    import joblib
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'ann_scaler.pkl'))
    
    # Create datasets and dataloaders
    train_dataset = DiabetesDataset(X_train, y_train)
    val_dataset = DiabetesDataset(X_val, y_val)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define model, loss function, and optimizer
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    model = DiabetesANN(input_size=input_size, num_classes=num_classes)
    model = model.to(device)
    
    # Display model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Use Cross Entropy Loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("\nStarting model training...")
    trained_model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        patience=10
    )
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_ann_model.pt')))
    
    # Evaluate the model
    class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
    results = evaluate_model(model, test_loader, criterion, device, class_names)
    
    # Plot training history and confusion matrix
    plot_training_history(history, save_path=os.path.join(REPORTS_DIR, 'ann_training_history.png'))
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         save_path=os.path.join(REPORTS_DIR, 'ann_confusion_matrix.png'))
    
    # Save model information
    save_model_info(
        model=model,
        input_size=input_size,
        feature_names=selected_features,
        class_names=class_names,
        results=results,
        save_path=os.path.join(REPORTS_DIR, 'ann_model_info.txt')
    )
    
    # Save full model with architecture (for deployment)
    torch.save(model, os.path.join(MODELS_DIR, 'ann_model_full.pt'))
    print(f"Full model saved to {os.path.join(MODELS_DIR, 'ann_model_full.pt')}")
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 