#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of wrapper methods for feature selection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed to ensure reproducibility
RANDOM_STATE = 42

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CURRENT_DIR, 'figures')

def forward_selection(X_train, y_train, feature_names, max_features=15):
    """
    Implement forward feature selection
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        feature_names: List of feature names
        max_features: Maximum number of features to select
        
    Returns:
        list: List of selected features
    """
    print("\n=== Forward Feature Selection ===")
    
    # Use cross-validation to evaluate each feature combination
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Check if it's a multiclass problem
    is_multiclass = len(np.unique(y_train)) > 2
    scoring = 'accuracy' if is_multiclass else 'roc_auc'
    
    # Try to use GPU acceleration (only if CUDA environment is available)
    try:
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        has_gpu = True
        print("GPU acceleration available, will use cuML")
    except (ImportError, ModuleNotFoundError):
        has_gpu = False
        print("GPU acceleration not available, using CPU")
    
    # Initialize
    selected_features = []
    remaining_features = feature_names.copy()
    current_score = 0
    
    # Save performance history by feature count
    performance_history = []
    
    # Forward selection process
    for i in range(min(max_features, len(feature_names))):
        best_feature = None
        best_score = current_score  # Change: use current score as baseline, not 0
        
        # Try each remaining feature
        for feature in remaining_features:
            # Current feature set
            current_features = selected_features + [feature]
            
            # Evaluate with logistic regression model
            if has_gpu:
                # GPU version of logistic regression
                lr = cuLogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[current_features].values
                scores = cross_val_score(lr, X_current, y_train, 
                                        cv=cv, scoring=scoring)
            else:
                # CPU version
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                scores = cross_val_score(lr, X_train[current_features], y_train, 
                                        cv=cv, scoring=scoring)
            
            avg_score = np.mean(scores)
            
            # If this feature improves performance, save it
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature
        
        # If a performance-improving feature was found, add it
        if best_feature and best_score > current_score:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            current_score = best_score
            performance_history.append((len(selected_features), current_score, selected_features.copy()))
            print(f"Added feature '{best_feature}', {scoring.upper()} = {current_score:.4f}")
        else:
            # If no feature improves performance, stop
            print("No more features improve performance, stopping selection.")
            break
    
    # Ensure at least some features are selected (even if performance doesn't improve)
    if len(selected_features) == 0:
        print("Forward selection didn't select any features. Selecting top 5 individually highest scoring features.")
        
        # Evaluate each individual feature's performance
        feature_scores = []
        for feature in feature_names:
            if has_gpu:
                lr = cuLogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]].values
            else:
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]]
                
            scores = cross_val_score(lr, X_current, y_train, cv=cv, scoring=scoring)
            feature_scores.append((feature, np.mean(scores)))
        
        # Sort by score
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 5 features
        selected_features = [f[0] for f in feature_scores[:5]]
        print(f"Top 5 individually highest scoring features: {selected_features}")
    
    # Visualize forward selection process
    plt.figure(figsize=(10, 6))
    
    if len(performance_history) > 0:
        features_count = [item[0] for item in performance_history]
        scores = [item[1] for item in performance_history]
        
        plt.plot(features_count, scores, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel(f'{scoring.upper()} Score')
        plt.title('Forward Feature Selection: Performance vs Number of Features')
        plt.grid(True)
    else:
        # If using alternate method for feature selection
        plt.text(0.5, 0.5, 'Used individual feature scores for selection', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'forward_selection.png'))
    plt.close()
    
    print(f"Final feature set from forward selection ({len(selected_features)} features): {selected_features}")
    
    return selected_features

def backward_elimination(X_train, y_train, feature_names):
    """
    Implement backward feature elimination
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        feature_names: List of feature names
        
    Returns:
        list: List of selected features
    """
    print("\n=== Backward Feature Elimination ===")
    
    # Use cross-validation to evaluate each feature combination
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Check if it's a multiclass problem
    is_multiclass = len(np.unique(y_train)) > 2
    scoring = 'accuracy' if is_multiclass else 'roc_auc'
    
    # Try to use GPU acceleration (only if CUDA environment is available)
    try:
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        has_gpu = True
        print("GPU acceleration available, will use cuML")
    except (ImportError, ModuleNotFoundError):
        has_gpu = False
        print("GPU acceleration not available, using CPU")
    
    # Initialize - to speed up, first limit to top 15 most relevant features 
    if len(feature_names) > 15:
        print(f"For faster computation, pre-filtering (selecting top 15 from {len(feature_names)} features)")
        # Pre-filtering - use individual feature scores
        feature_scores = []
        for feature in feature_names:
            if has_gpu:
                lr = cuLogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]].values
            else:
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]]
                
            scores = cross_val_score(lr, X_current, y_train, cv=3, scoring=scoring)
            feature_scores.append((feature, np.mean(scores)))
        
        # Sort by score, select top 15 to start backward elimination
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in feature_scores[:15]]
        print(f"Pre-filtered features: {selected_features}")
    else:
        selected_features = feature_names.copy()
    
    # Calculate initial performance with all features
    if has_gpu:
        lr = cuLogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        X_current = X_train[selected_features].values
        scores = cross_val_score(lr, X_current, y_train, cv=cv, scoring=scoring)
    else:
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        scores = cross_val_score(lr, X_train[selected_features], y_train, cv=cv, scoring=scoring)
    
    current_score = np.mean(scores)
    
    print(f"Initial performance (using {len(selected_features)} features): {scoring.upper()} = {current_score:.4f}")
    
    # Save performance history by feature count
    performance_history = [(len(selected_features), current_score, selected_features.copy())]
    
    # Optimization: use batch evaluation method
    max_iterations = min(10, len(selected_features) - 1)  # Limit number of iterations
    for iteration in range(max_iterations):
        if len(selected_features) <= 5:  # Stop when 5 or fewer features remain
            break
            
        print(f"Iteration {iteration+1}/{max_iterations}, current feature count: {len(selected_features)}")
        
        # Calculate importance of each feature
        feature_importance = {}
        
        # Evaluate by removing each feature one at a time
        for feature in selected_features:
            temp_features = [f for f in selected_features if f != feature]
            
            if has_gpu:
                X_current = X_train[temp_features].values
                scores = cross_val_score(lr, X_current, y_train, cv=cv, scoring=scoring)
            else:
                scores = cross_val_score(lr, X_train[temp_features], y_train, cv=cv, scoring=scoring)
                
            avg_score = np.mean(scores)
            feature_importance[feature] = avg_score
        
        # Find features with highest score when removed
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Remove top 20% scoring features
        n_to_remove = max(1, int(len(selected_features) * 0.2))
        features_to_remove = [f[0] for f in sorted_features[:n_to_remove]]
        
        # Update feature set and score
        for feature in features_to_remove:
            selected_features.remove(feature)
            
        # Evaluate new feature set
        if has_gpu:
            X_current = X_train[selected_features].values
            scores = cross_val_score(lr, X_current, y_train, cv=cv, scoring=scoring)
        else:
            scores = cross_val_score(lr, X_train[selected_features], y_train, cv=cv, scoring=scoring)
            
        current_score = np.mean(scores)
        
        print(f"After removing {n_to_remove} features: {scoring.upper()} = {current_score:.4f}")
        performance_history.append((len(selected_features), current_score, selected_features.copy()))
    
    # Select best feature set based on performance history
    best_score_idx = np.argmax([x[1] for x in performance_history])
    best_features = performance_history[best_score_idx][2]
    best_score = performance_history[best_score_idx][1]
    
    print(f"Best performance ({scoring.upper()}={best_score:.4f}) achieved with {len(best_features)} features")
    
    # Visualize backward elimination process
    plt.figure(figsize=(10, 6))
    features_count = [item[0] for item in performance_history]
    scores = [item[1] for item in performance_history]
    
    plt.plot(features_count, scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel(f'{scoring.upper()} Score')
    plt.title('Backward Feature Elimination: Performance vs Number of Features')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'backward_elimination.png'))
    plt.close()
    
    print(f"Final feature set from backward elimination ({len(best_features)} features): {best_features}")
    
    return best_features

def recursive_feature_elimination(X_train, y_train, X_test, y_test, feature_names):
    """
    Feature selection using Recursive Feature Elimination (RFE)
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        X_test: Test set features
        y_test: Test set target variable
        feature_names: List of feature names
        
    Returns:
        dict: RFE feature selection results
    """
    print("\n=== Recursive Feature Elimination (RFE) ===")
    results = {}
    
    # RFE with Logistic Regression
    print("\nRFE with Logistic Regression:")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rfe_lr = RFE(estimator=lr, n_features_to_select=10, step=1)
    rfe_lr.fit(X_train, y_train)
    
    # Get feature rankings
    lr_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe_lr.ranking_,
        'Selected': rfe_lr.support_
    })
    lr_ranking = lr_ranking.sort_values('Ranking')
    results['lr_ranking'] = lr_ranking
    
    # Output selected features
    selected_features_lr = lr_ranking[lr_ranking['Selected']]['Feature'].tolist()
    print(f"Features selected by Logistic Regression RFE: {selected_features_lr}")
    
    # Evaluate the performance of the selected feature set
    X_train_selected = X_train[selected_features_lr]
    X_test_selected = X_test[selected_features_lr]
    
    lr.fit(X_train_selected, y_train)
    lr_pred = lr.predict(X_test_selected)
    lr_prob = lr.predict_proba(X_test_selected)[:, 1]
    
    acc_lr = accuracy_score(y_test, lr_pred)
    auc_lr = roc_auc_score(y_test, lr_prob)
    
    print(f"Logistic Regression performance with RFE-selected features: Accuracy={acc_lr:.4f}, AUC={auc_lr:.4f}")
    
    # RFE with Random Forest
    print("\nRFE with Random Forest:")
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rfe_rf = RFE(estimator=rf, n_features_to_select=10, step=1)
    rfe_rf.fit(X_train, y_train)
    
    # Get feature rankings
    rf_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe_rf.ranking_,
        'Selected': rfe_rf.support_
    })
    rf_ranking = rf_ranking.sort_values('Ranking')
    results['rf_ranking'] = rf_ranking
    
    # Output selected features
    selected_features_rf = rf_ranking[rf_ranking['Selected']]['Feature'].tolist()
    print(f"Features selected by Random Forest RFE: {selected_features_rf}")
    
    # Evaluate the performance of the selected feature set
    X_train_selected = X_train[selected_features_rf]
    X_test_selected = X_test[selected_features_rf]
    
    rf.fit(X_train_selected, y_train)
    rf_pred = rf.predict(X_test_selected)
    rf_prob = rf.predict_proba(X_test_selected)[:, 1]
    
    acc_rf = accuracy_score(y_test, rf_pred)
    auc_rf = roc_auc_score(y_test, rf_prob)
    
    print(f"Random Forest performance with RFE-selected features: Accuracy={acc_rf:.4f}, AUC={auc_rf:.4f}")
    
    # Visualize RFE results
    plt.figure(figsize=(12, 8))
    
    # Combine logistic regression and random forest results for comparison
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'LR Ranking': rfe_lr.ranking_,
        'RF Ranking': rfe_rf.ranking_
    })
    comparison_df = comparison_df.sort_values('LR Ranking')
    
    bar_width = 0.35
    r1 = np.arange(len(feature_names))
    r2 = [x + bar_width for x in r1]
    
    plt.barh(r1, comparison_df['LR Ranking'], height=bar_width, label='Logistic Regression Ranking')
    plt.barh(r2, comparison_df['RF Ranking'], height=bar_width, label='Random Forest Ranking')
    
    plt.yticks([r + bar_width/2 for r in range(len(feature_names))], comparison_df['Feature'])
    plt.xlabel('Ranking (Lower is Better)')
    plt.title('Feature Rankings Comparison: LR vs RF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rfe_comparison.png'))
    plt.close()
    
    return results, selected_features_lr, selected_features_rf

def compare_cv_results(X_train, y_train, feature_subsets, subset_names):
    """
    Compare cross-validation results for different feature subsets
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        feature_subsets: Dictionary of feature subsets (name: feature list)
        subset_names: List of subset names
        
    Returns:
        DataFrame: Comparison results
    """
    print("\n=== Feature Subset Cross-Validation Comparison ===")
    
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Use cross-validation to evaluate subsets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Different models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    # Store results
    results = []
    
    for name, features in feature_subsets.items():
        if not features:
            continue
            
        print(f"\nEvaluating feature subset: {name} ({len(features)} features)")
        
        for model_name, model in models.items():
            # Calculate cross-validation scores
            scores = cross_val_score(model, X_train[features], y_train, cv=cv, scoring='roc_auc')
            
            # Save results
            results.append({
                'Feature Subset': name,
                'Model': model_name,
                'Num Features': len(features),
                'Mean AUC': np.mean(scores),
                'Std AUC': np.std(scores)
            })
            
            print(f"  {model_name}: Mean AUC = {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize comparison results
    plt.figure(figsize=(14, 8))
    
    # Plot grouped bar chart for each model
    bar_width = 0.35
    models_list = results_df['Model'].unique()
    
    for i, model in enumerate(models_list):
        model_data = results_df[results_df['Model'] == model]
        x = np.arange(len(model_data))
        offset = i * bar_width
        
        plt.bar(x + offset, model_data['Mean AUC'], 
                width=bar_width, 
                label=model,
                yerr=model_data['Std AUC'],
                capsize=5)
    
    # Set chart properties
    plt.xlabel('Feature Subset')
    plt.ylabel('AUC Score')
    plt.title('Model Performance Comparison Across Feature Subsets')
    plt.xticks(np.arange(len(results_df['Feature Subset'].unique())) + bar_width/2, 
              results_df['Feature Subset'].unique(), rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_subset_comparison.png'))
    plt.close()
    
    return results_df

def perform_rfe_selection(X_train, y_train, feature_names, n_features=10):
    """
    Perform Recursive Feature Elimination and return selected features
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        feature_names: List of feature names
        n_features: Number of features to select
        
    Returns:
        list: Selected features list
    """
    print("Performing Recursive Feature Elimination (RFE)...")
    
    # Use logistic regression as base model
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='liblinear')
    
    # Check if it's a multiclass problem
    is_multiclass = len(np.unique(y_train)) > 2
    
    # Choose scoring metric based on problem type
    if is_multiclass:
        scoring = 'accuracy'
        print("Target variable is multiclass, using accuracy as scoring metric")
    else:
        scoring = 'roc_auc'
        print("Target variable is binary, using AUC as scoring metric")
    
    # Use RFECV to find optimal number of features
    rfecv = RFECV(
        estimator=lr,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring,
        min_features_to_select=5
    )
    rfecv.fit(X_train, y_train)
    
    optimal_n_features = rfecv.n_features_
    print(f"Optimal number of features determined by RFECV: {optimal_n_features}")
    
    # Run RFE with the determined optimal number of features
    rfe = RFE(estimator=lr, n_features_to_select=min(optimal_n_features, n_features), step=1)
    rfe.fit(X_train, y_train)
    
    # Get selected features
    selected_features = [feature for feature, selected in zip(feature_names, rfe.support_) if selected]
    
    # Visualize feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rfe.ranking_
    })
    feature_importance = feature_importance.sort_values('Importance')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('RFE Feature Ranking (Lower is Better)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rfe_feature_ranking.png'))
    plt.close()
    
    print(f"Features selected by RFE ({len(selected_features)}): {selected_features}")
    return selected_features

def perform_sequential_selection(X_train, y_train, feature_names, max_features=10):
    """
    Perform sequential feature selection (combination of forward selection and backward elimination)
    
    Args:
        X_train: Training set features
        y_train: Training set target variable
        feature_names: List of feature names
        max_features: Maximum number of features to select
        
    Returns:
        list: Selected features list
    """
    print("Performing sequential feature selection...")
    
    # First run forward selection
    forward_features = forward_selection(X_train, y_train, feature_names, max_features)
    
    # If forward selection found a good enough feature subset (at least 5), skip backward elimination to save time
    if len(forward_features) >= 5:
        print(f"Forward selection found {len(forward_features)} features, skipping backward elimination step")
        return forward_features
    
    # Otherwise continue with backward elimination and merge results
    backward_features = backward_elimination(X_train, y_train, feature_names)
    
    # Combine and deduplicate
    all_features = list(set(forward_features + backward_features))
    
    # If combined features are too many, keep the most important ones up to max_features
    if len(all_features) > max_features:
        # Check if it's a multiclass problem
        is_multiclass = len(np.unique(y_train)) > 2
        scoring = 'accuracy' if is_multiclass else 'roc_auc'
        
        # Use cross-validation to evaluate each feature's individual contribution
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        feature_scores = []
        
        # Try to use GPU acceleration
        try:
            from cuml.linear_model import LogisticRegression as cuLogisticRegression
            has_gpu = True
        except (ImportError, ModuleNotFoundError):
            has_gpu = False
        
        for feature in all_features:
            if has_gpu:
                lr = cuLogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]].values
            else:
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                X_current = X_train[[feature]]
                
            scores = cross_val_score(lr, X_current, y_train, cv=cv, scoring=scoring)
            feature_scores.append((feature, np.mean(scores)))
        
        # Sort by importance
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in feature_scores[:max_features]]
    else:
        selected_features = all_features
    
    print(f"Sequential feature selection results ({len(selected_features)}): {selected_features}")
    return selected_features

if __name__ == "__main__":
    # This file is typically called from the main file, not run independently
    print("This file contains wrapper methods for feature selection and should be called from the main program.") 