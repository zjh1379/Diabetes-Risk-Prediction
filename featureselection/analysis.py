#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis and integration of feature selection results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set random seed to ensure reproducibility
RANDOM_STATE = 42

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CURRENT_DIR, 'figures')
REPORTS_DIR = os.path.join(CURRENT_DIR, 'reports')

def consolidate_feature_selection(all_feature_sets, feature_names):
    """
    Comprehensive analysis of results from various feature selection methods
    
    Args:
        all_feature_sets: Dictionary of feature sets selected by different methods
        feature_names: List of all feature names
        
    Returns:
        DataFrame: Consolidated feature scores
        list: Multiple candidate feature subsets
    """
    print("\n=== Comprehensive Analysis of Feature Selection Results ===")
    
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Create feature scoring framework
    feature_scores = pd.DataFrame({'Feature': feature_names})
    feature_scores['Selected_Count'] = 0
    
    # Calculate how many times each feature was selected
    for method_name, feature_set in all_feature_sets.items():
        # Add each method's selection result as a column
        feature_scores[f'Selected_by_{method_name}'] = feature_scores['Feature'].isin(feature_set).astype(int)
        # Accumulate selection count
        feature_scores['Selected_Count'] += feature_scores[f'Selected_by_{method_name}']
    
    # Sort by selection count
    feature_scores = feature_scores.sort_values('Selected_Count', ascending=False)
    
    print("\nFeature selection frequency across methods:")
    print(feature_scores[['Feature', 'Selected_Count']].head(15))
    
    # Visualize feature selection frequency
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Selected_Count', y='Feature', data=feature_scores.head(15))
    plt.title('Features Selection Frequency Across Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_selection_frequency.png'))
    plt.close()
    
    # Generate candidate feature subsets
    candidate_subsets = {}
    
    # Subset 1: Features selected by majority of methods (at least half)
    threshold = len(all_feature_sets) / 2
    consensus_features = feature_scores[feature_scores['Selected_Count'] >= threshold]['Feature'].tolist()
    candidate_subsets['Consensus'] = consensus_features
    
    # Subset 2: Features selected by all methods
    all_selected = feature_scores[feature_scores['Selected_Count'] == len(all_feature_sets)]['Feature'].tolist()
    if all_selected:
        candidate_subsets['All_Methods'] = all_selected
    
    # Subset 3: Top N most frequently selected features
    top_n = 10
    top_features = feature_scores.head(top_n)['Feature'].tolist()
    candidate_subsets['Top_10'] = top_features
    
    # Print candidate feature subsets
    print("\nCandidate feature subsets:")
    for name, features in candidate_subsets.items():
        print(f"{name} ({len(features)} features): {features}")
    
    return feature_scores, candidate_subsets

def stability_analysis(X, y, feature_subsets, n_splits=5):
    """
    Validate feature selection results using different data splits
    
    Args:
        X: Feature data
        y: Target variable
        feature_subsets: Dictionary of feature subsets
        n_splits: Number of cross-validation folds
        
    Returns:
        DataFrame: Stability analysis results
    """
    print("\n=== Feature Selection Stability Analysis ===")
    
    # Create different data splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Check if it's a multiclass problem
    is_multiclass = len(np.unique(y)) > 2
    
    # For storing results
    stability_results = []
    
    # Analyze each feature subset
    for subset_name, features in feature_subsets.items():
        print(f"\nAnalyzing feature subset: {subset_name}")
        
        fold_results = []
        
        # Train and evaluate on each fold
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Evaluate using logistic regression and random forest
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE)
            }
            
            for model_name, model in models.items():
                # Train and evaluate model
                model.fit(X_train[features], y_train)
                y_pred = model.predict(X_test[features])
                
                # Calculate accuracy and F1 score
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Calculate AUC for binary classification
                if is_multiclass:
                    auc_score = 0.0
                else:
                    y_prob = model.predict_proba(X_test[features])[:, 1]
                    auc_score = roc_auc_score(y_test, y_prob)
                
                # Save results
                fold_results.append({
                    'Subset': subset_name,
                    'Model': model_name,
                    'Fold': fold + 1,
                    'Accuracy': acc,
                    'AUC': auc_score,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1
                })
        
        # Calculate and print average performance for each fold
        fold_df = pd.DataFrame(fold_results)
        
        for model_name in models.keys():
            model_results = fold_df[fold_df['Model'] == model_name]
            mean_results = model_results.mean(numeric_only=True)
            std_results = model_results.std(numeric_only=True)
            
            print(f"{model_name} average performance ({n_splits}-fold cross-validation):")
            print(f"  Accuracy: {mean_results['Accuracy']:.4f} (±{std_results['Accuracy']:.4f})")
            
            if not is_multiclass:
                print(f"  AUC: {mean_results['AUC']:.4f} (±{std_results['AUC']:.4f})")
                
            print(f"  Precision: {mean_results['Precision']:.4f} (±{std_results['Precision']:.4f})")
            print(f"  Recall: {mean_results['Recall']:.4f} (±{std_results['Recall']:.4f})")
            print(f"  F1 Score: {mean_results['F1']:.4f} (±{std_results['F1']:.4f})")
            
            # Add summary to overall results
            stability_results.append({
                'Subset': subset_name,
                'Model': model_name,
                'Mean_Accuracy': mean_results['Accuracy'],
                'Std_Accuracy': std_results['Accuracy'],
                'Mean_AUC': mean_results['AUC'] if not is_multiclass else 0.0,
                'Std_AUC': std_results['AUC'] if not is_multiclass else 0.0,
                'Mean_F1': mean_results['F1'],
                'Std_F1': std_results['F1']
            })
    
    # Convert results to DataFrame
    stability_df = pd.DataFrame(stability_results)
    
    # Visualize stability results
    plt.figure(figsize=(14, 10))
    
    # Plot comparison of accuracy for different subsets for each model
    bar_width = 0.35
    models = stability_df['Model'].unique()
    subsets = stability_df['Subset'].unique()
    
    x = np.arange(len(subsets))
    
    # Use accuracy for multiclass, AUC for binary
    metric = 'Mean_Accuracy' if is_multiclass else 'Mean_AUC'
    error_metric = 'Std_Accuracy' if is_multiclass else 'Std_AUC'
    metric_label = 'Accuracy' if is_multiclass else 'AUC Score'
    
    for i, model in enumerate(models):
        model_data = stability_df[stability_df['Model'] == model]
        
        plt.bar(x + i*bar_width, model_data[metric], 
               width=bar_width, 
               label=model,
               yerr=model_data[error_metric],
               capsize=5)
    
    plt.xlabel('Feature Subsets')
    plt.ylabel(metric_label)
    plt.title('Model Performance Stability Comparison Across Feature Subsets')
    plt.xticks(x + bar_width/2, subsets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'stability_analysis.png'))
    plt.close()
    
    return stability_df

def learning_curves(X, y, feature_subsets):
    """
    Generate learning curves for different feature subsets
    
    Args:
        X: Feature data
        y: Target variable
        feature_subsets: Dictionary of feature subsets
    """
    print("\n=== Generating Learning Curves ===")
    
    # Check if it's a multiclass problem
    is_multiclass = len(np.unique(y)) > 2
    
    # Training set proportions for validation curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    fig, axes = plt.subplots(len(feature_subsets), 2, figsize=(15, 5 * len(feature_subsets)))
    
    # Use logistic regression and random forest
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    # Generate learning curves for each feature subset
    for i, (subset_name, features) in enumerate(feature_subsets.items()):
        print(f"\nGenerating learning curves for feature subset: {subset_name}")
        
        for j, (model_name, model) in enumerate(models.items()):
            # Select the correct subplot
            ax = axes[i, j] if len(feature_subsets) > 1 else axes[j]
            
            # Choose scoring metric based on problem type
            if is_multiclass:
                scoring = 'accuracy'  # For multiclass, use accuracy
            else:
                try:
                    # For binary classification, try to use ROC AUC
                    scoring = 'roc_auc'
                    
                    # Make sure model has predict_proba method needed for AUC calculation
                    if not hasattr(model, "predict_proba"):
                        scoring = 'accuracy'
                        print(f"Warning: {model_name} doesn't support predict_proba, using accuracy instead of AUC.")
                except Exception as e:
                    scoring = 'accuracy'
                    print(f"Warning: Using accuracy instead of AUC due to error: {str(e)}")
            
            try:
                # Calculate learning curve
                train_sizes_, train_scores, test_scores = learning_curve(
                    model, X[features], y, 
                    train_sizes=train_sizes, 
                    cv=5, 
                    scoring=scoring,
                    random_state=RANDOM_STATE,
                    n_jobs=-1)
                
                # Calculate mean and standard deviation
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                # Plot learning curve
                ax.plot(train_sizes_, train_mean, 'o-', color='blue', label='Training Score')
                ax.plot(train_sizes_, test_mean, 'o-', color='red', label='Validation Score')
                
                # Plot error bands
                ax.fill_between(train_sizes_, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                ax.fill_between(train_sizes_, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
                
                # Add legend and title
                ax.set_title(f'{subset_name} - {model_name}')
                ax.set_xlabel('Training Samples')
                score_label = 'Score' if scoring == 'accuracy' else 'AUC Score'
                ax.set_ylabel(score_label)
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.5)
                
            except Exception as e:
                print(f"Error generating learning curve for {subset_name} with {model_name}: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'learning_curves.png'))
    plt.close()

def generate_final_report(feature_scores, stability_results, best_subset, best_features):
    """
    Generate final feature selection report
    
    Args:
        feature_scores: Consolidated feature scores
        stability_results: Stability analysis results
        best_subset: Name of best feature subset
        best_features: List of best features
        
    Returns:
        None
    """
    print("\n=== Feature Selection Final Report ===")
    
    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diabetes Risk Prediction - Feature Selection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .highlight {{ background-color: #e6f7ff; font-weight: bold; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diabetes Risk Prediction - Feature Selection Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Through comprehensive analysis of multiple feature selection methods, we have determined the best feature subset.</p>
                <h3>Best Feature Subset: {best_subset}</h3>
                <p>Contains {len(best_features)} features: {', '.join(best_features)}</p>
            </div>
            
            <h2>Feature Selection Frequency</h2>
            <p>The table below shows how frequently each feature was selected across different feature selection methods:</p>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Selection Count</th>
                </tr>
    """
    
    # Add feature selection frequency table
    for _, row in feature_scores.head(15).iterrows():
        highlight = 'class="highlight"' if row['Feature'] in best_features else ''
        html_content += f"""
                <tr {highlight}>
                    <td>{row['Feature']}</td>
                    <td>{row['Selected_Count']}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Stability Analysis Results</h2>
            <p>The table below shows performance comparison across different feature subsets and models:</p>
            <table>
                <tr>
                    <th>Feature Subset</th>
                    <th>Model</th>
                    <th>Mean AUC</th>
                    <th>AUC Std Dev</th>
                    <th>Mean F1 Score</th>
                </tr>
    """
    
    # Add stability analysis results
    for _, row in stability_results.iterrows():
        highlight = 'class="highlight"' if row['Subset'] == best_subset else ''
        html_content += f"""
                <tr {highlight}>
                    <td>{row['Subset']}</td>
                    <td>{row['Model']}</td>
                    <td>{row['Mean_AUC']:.4f}</td>
                    <td>{row['Std_AUC']:.4f}</td>
                    <td>{row['Mean_F1']:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Feature Selection Conclusions</h2>
            <p>Based on comprehensive analysis, we draw the following conclusions:</p>
            <ul>
                <li>The most important features are primarily related to health indicators and lifestyle factors</li>
                <li>Selecting an appropriate feature subset can improve model performance and reduce overfitting</li>
                <li>These features are key indicators for predicting diabetes risk</li>
            </ul>
            
            <h2>Charts</h2>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                <div style="width: 48%; margin-bottom: 20px;">
                    <img src="../figures/feature_selection_frequency.png" style="width: 100%;">
                    <p>Feature Selection Frequency</p>
                </div>
                <div style="width: 48%; margin-bottom: 20px;">
                    <img src="../figures/stability_analysis.png" style="width: 100%;">
                    <p>Stability Analysis</p>
                </div>
                <div style="width: 48%; margin-bottom: 20px;">
                    <img src="../figures/learning_curves.png" style="width: 100%;">
                    <p>Learning Curves Analysis</p>
                </div>
                <div style="width: 48%; margin-bottom: 20px;">
                    <img src="../figures/feature_subset_comparison.png" style="width: 100%;">
                    <p>Feature Subset Comparison</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report - using UTF-8 encoding to solve Unicode errors
    with open(os.path.join(REPORTS_DIR, 'feature_selection_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save best features list to CSV
    best_features_df = pd.DataFrame({'Feature': best_features})
    best_features_df.to_csv(os.path.join(REPORTS_DIR, 'best_features.csv'), index=False)
    
    print("\nFinal report has been generated, see reports/feature_selection_report.html")
    print(f"Best feature set ({len(best_features)} features) has been saved to reports/best_features.csv")

if __name__ == "__main__":
    # This file is typically called from the main file, not run independently
    print("This file contains feature selection analysis functionality and should be called from the main program.") 