#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for feature selection using a real dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set default font to SimHei
matplotlib.rcParams['axes.unicode_minus'] = False    # Fix issue with minus sign display in saved figures

# Current project path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create results directory
result_dir = os.path.join(PROJECT_DIR, 'test_results')
os.makedirs(result_dir, exist_ok=True)

# Set random seed to ensure reproducibility
RANDOM_STATE = 42

# Load real dataset
print("Loading real dataset...")
data_path = os.path.join(PROJECT_DIR, 'data', 'diabetes_012_health_indicators_BRFSS2015.csv', 
                        'diabetes_012_health_indicators_BRFSS2015.csv')

# Read data
df = pd.read_csv(data_path)
print(f"Dataset size: {df.shape[0]} samples, {df.shape[1]} columns")

# Basic data exploration
print("\nDataset information:")
print(df.dtypes)

print("\nTarget variable distribution:")
target_counts = df['Diabetes_012'].value_counts()
print(target_counts)
print(f"Target variable percentages: \n{(target_counts / len(df) * 100).round(2)}%")

# Visualize target variable distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Diabetes_012', data=df)
plt.title('Diabetes Distribution')
plt.xlabel('Diabetes Status (0: No Diabetes, 1: Prediabetes, 2: Diabetes)')
plt.ylabel('Count')

# Add value labels
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
               (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'bottom', 
               xytext = (0, 5), textcoords = 'offset points')

plt.savefig(os.path.join(result_dir, 'target_distribution.png'))
plt.close()

# Prepare data
print("\nPreparing data for feature selection...")
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']
feature_names = X.columns.tolist()

print(f"Number of features: {len(feature_names)}")
print(f"Feature list: {feature_names}")

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Use Random Forest for feature importance calculation
print("\nCalculating Random Forest feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# Get feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance ranking (top 15 features):")
print(importances.head(15))

# Select top 10 features
selected_features = importances.head(10)['feature'].tolist()
print(f"\nSelected best feature set: {selected_features}")

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importances.head(15))
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'feature_importance.png'))
print(f"\nFeature importance chart saved to {os.path.join(result_dir, 'feature_importance.png')}")

# Evaluate model performance with selected features
print("\nEvaluating model performance with selected features...")
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train and evaluate using all features
rf_all = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_all.fit(X_train, y_train)
y_pred_all = rf_all.predict(X_test)

# Train and evaluate using selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_selected.fit(X_train_selected, y_train)
y_pred_selected = rf_selected.predict(X_test_selected)

# Calculate accuracy and F1 score
acc_all = accuracy_score(y_test, y_pred_all)
f1_all = f1_score(y_test, y_pred_all, average='weighted')

acc_selected = accuracy_score(y_test, y_pred_selected)
f1_selected = f1_score(y_test, y_pred_selected, average='weighted')

# Print performance comparison
print("\nModel performance comparison:")
print(f"All features ({X.shape[1]}): Accuracy = {acc_all:.4f}, F1 score = {f1_all:.4f}")
print(f"Selected features ({len(selected_features)}): Accuracy = {acc_selected:.4f}, F1 score = {f1_selected:.4f}")

# Save selected features to CSV
selected_features_df = pd.DataFrame({'feature': selected_features})
selected_features_df.to_csv(os.path.join(result_dir, 'selected_features.csv'), index=False)
print(f"\nSelected features saved to {os.path.join(result_dir, 'selected_features.csv')}")

# Visualize performance comparison
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'F1 Score']
all_scores = [acc_all, f1_all]
selected_scores = [acc_selected, f1_selected]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, all_scores, width, label=f'All Features ({X.shape[1]})')
plt.bar(x + width/2, selected_scores, width, label=f'Selected Features ({len(selected_features)})')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Performance: All Features vs Selected Features')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'performance_comparison.png'))
print(f"\nPerformance comparison chart saved to {os.path.join(result_dir, 'performance_comparison.png')}")

# Output feature analysis summary
print("\nFeature selection analysis summary:")
print(f"1. Most important feature: {selected_features[0]} (importance value: {importances.iloc[0]['importance']:.4f})")
print(f"2. Selecting the top 10 features achieves similar performance to using all {X.shape[1]} features")
print(f"   - Accuracy: {acc_selected:.4f} vs {acc_all:.4f} (difference: {abs(acc_selected-acc_all):.4f})")
print(f"   - F1 score: {f1_selected:.4f} vs {f1_all:.4f} (difference: {abs(f1_selected-f1_all):.4f})")
print(f"3. Feature reduction: {X.shape[1] - len(selected_features)} features ({(1 - len(selected_features)/X.shape[1])*100:.1f}%)")
print("\nFeature selection analysis completed!") 