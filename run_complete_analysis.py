#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for complete analysis of diabetes dataset using the feature selection package
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set default font to SimHei
matplotlib.rcParams['axes.unicode_minus'] = False    # Fix issue with minus sign display in saved figures

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'feature_selection_output')

# Ensure Python can find project modules
sys.path.insert(0, CURRENT_DIR)

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)

# Import feature selection modules
from featureselection.feature_selection import (
    load_and_preprocess_data,
    perform_filter_selection
)

from featureselection.wrapper_methods import (
    perform_rfe_selection,
    perform_sequential_selection
)

from featureselection.embedded_methods import (
    perform_lasso_selection,
    perform_rf_selection,
    perform_xgboost_selection
)

from featureselection.analysis import (
    consolidate_feature_selection,
    stability_analysis,
    learning_curves,
    generate_final_report
)

# Set random seed to ensure reproducibility
RANDOM_STATE = 42

def main():
    """Main function to run the complete feature selection process"""
    print("===== Diabetes Risk Prediction Feature Selection Analysis =====")
    
    # Ensure output directories exist
    for output_path in ['figures', 'reports']:
        os.makedirs(os.path.join(OUTPUT_DIR, output_path), exist_ok=True)
        
    # Replace output paths in modules
    import featureselection.analysis as analysis
    analysis.FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    analysis.REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
    
    import featureselection.feature_selection as feature_selection
    feature_selection.FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    
    import featureselection.wrapper_methods as wrapper_methods
    wrapper_methods.FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    
    import featureselection.embedded_methods as embedded_methods
    embedded_methods.FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    
    # 1. Data loading and preprocessing
    print("\n1. Data Loading and Preprocessing")
    data_path = os.path.join(DATA_DIR, 'diabetes_012_health_indicators_BRFSS2015.csv', 
                           'diabetes_012_health_indicators_BRFSS2015.csv')
    X, y, feature_names = load_and_preprocess_data(data_path)
    print(f"* Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"* Feature list: {feature_names}")
    
    # Split dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"* Training set size: {X_train.shape[0]} samples")
    print(f"* Test set size: {X_test.shape[0]} samples")
    
    # 2. Run various feature selection methods
    print("\n2. Running Feature Selection Methods")
    
    # Store results from different methods
    all_feature_sets = {}
    
    # 2.1 Filter-based feature selection
    print("\n2.1 Filter-based Feature Selection")
    filter_selected = perform_filter_selection(X_train, y_train, feature_names)
    all_feature_sets["Filter"] = filter_selected
    
    # 2.2 Wrapper-based feature selection
    print("\n2.2 Wrapper-based Feature Selection")
    rfe_selected = perform_rfe_selection(X_train, y_train, feature_names)
    all_feature_sets["RFE"] = rfe_selected
    
    seq_selected = perform_sequential_selection(X_train, y_train, feature_names)
    all_feature_sets["Sequential"] = seq_selected
    
    # 2.3 Embedded-based feature selection
    print("\n2.3 Embedded-based Feature Selection")
    lasso_selected = perform_lasso_selection(X_train, y_train, feature_names)
    all_feature_sets["LASSO"] = lasso_selected
    
    rf_selected = perform_rf_selection(X_train, y_train, feature_names)
    all_feature_sets["RandomForest"] = rf_selected
    
    xgb_selected = perform_xgboost_selection(X_train, y_train, feature_names)
    all_feature_sets["XGBoost"] = xgb_selected
    
    # 3. Comprehensive analysis of feature selection results
    print("\n3. Comprehensive Analysis of Feature Selection Results")
    feature_scores, candidate_subsets = consolidate_feature_selection(all_feature_sets, feature_names)
    
    # 4. Stability analysis of candidate feature subsets
    print("\n4. Feature Stability Analysis")
    stability_results = stability_analysis(X, y, candidate_subsets)
    
    # 5. Learning curve analysis
    print("\n5. Learning Curve Analysis")
    learning_curves(X, y, candidate_subsets)
    
    # 6. Determine the best feature subset
    print("\n6. Determining the Best Feature Subset")
    # Select the best subset based on AUC
    best_subset = stability_results.sort_values('Mean_AUC', ascending=False).iloc[0]['Subset']
    best_features = candidate_subsets[best_subset]
    
    print(f"* Best feature subset: {best_subset}")
    print(f"* Features included ({len(best_features)}): {best_features}")
    
    # 7. Save feature comparison
    plt.figure(figsize=(14, 8))
    
    # Draw bar chart
    x = np.arange(len(candidate_subsets))
    width = 0.4
    
    # Calculate feature count for each subset
    subset_feature_counts = [len(features) for subset, features in candidate_subsets.items()]
    
    bars = plt.bar(x, subset_feature_counts, width, label='Feature Count')
    
    # Highlight the best subset
    best_idx = list(candidate_subsets.keys()).index(best_subset)
    bars[best_idx].set_color('red')
    
    # Set chart properties
    plt.title('Feature Subset Comparison')
    plt.xticks(x, candidate_subsets.keys(), rotation=45)
    plt.xlabel('Feature Subset')
    plt.ylabel('Number of Features')
    
    # Add text annotations
    for i, v in enumerate(subset_feature_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.axhline(y=X.shape[1], color='gray', linestyle='--', alpha=0.7, label='All Features')
    plt.text(len(subset_feature_counts)-1, X.shape[1] + 0.5, f'All Features: {X.shape[1]}', ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'feature_subset_comparison.png'))
    plt.close()
    
    # 8. Generate final report
    print("\n7. Generating Final Report")
    generate_final_report(feature_scores, stability_results, best_subset, best_features)
    
    print("\n===== Feature Selection Analysis Completed =====")
    print(f"Analysis results saved in {OUTPUT_DIR} directory")
    
    # Return the best feature set for subsequent model training
    return best_features

if __name__ == "__main__":
    best_features = main()
    
    # Save best features to file
    best_features_df = pd.DataFrame({'feature': best_features})
    best_features_df.to_csv(os.path.join(OUTPUT_DIR, 'best_features.csv'), index=False)
    print(f"Best feature set saved to {os.path.join(OUTPUT_DIR, 'best_features.csv')}") 