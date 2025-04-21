#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
嵌入法特征选择实现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可重复
RANDOM_STATE = 42

# 获取当前文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CURRENT_DIR, 'figures')

def perform_lasso_selection(X_train, y_train, feature_names, threshold=0.0):
    """
    执行LASSO特征选择并返回选择的特征
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        threshold: 系数阈值，高于此值的特征被选择
        
    Returns:
        list: 选择的特征列表
    """
    print("执行LASSO特征选择...")
    
    # 直接调用现有函数
    selected_features, _ = lasso_feature_selection(X_train, y_train, feature_names)
    
    return selected_features

def perform_rf_selection(X_train, y_train, feature_names, threshold=0.02):
    """
    执行随机森林特征选择并返回选择的特征
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        threshold: 重要性阈值，高于此值的特征被选择
        
    Returns:
        list: 选择的特征列表
    """
    print("执行随机森林特征选择...")
    
    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    # 计算特征重要性
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })
    rf_importance = rf_importance.sort_values('Importance', ascending=False)
    
    # 选择重要特征
    selected_features = rf_importance[rf_importance['Importance'] > threshold]['Feature'].tolist()
    
    # 如果选择的特征太少，则选择前10个特征
    if len(selected_features) < 5:
        selected_features = rf_importance.head(10)['Feature'].tolist()
    
    print(f"随机森林选择的特征 ({len(selected_features)}): {selected_features}")
    
    return selected_features

def perform_xgboost_selection(X_train, y_train, feature_names, threshold=0.02):
    """
    执行XGBoost特征选择并返回选择的特征
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        threshold: 重要性阈值，高于此值的特征被选择
        
    Returns:
        list: 选择的特征列表
    """
    print("执行XGBoost特征选择...")
    
    # 训练XGBoost模型
    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    
    # 计算特征重要性
    xgb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    })
    xgb_importance = xgb_importance.sort_values('Importance', ascending=False)
    
    # 选择重要特征
    selected_features = xgb_importance[xgb_importance['Importance'] > threshold]['Feature'].tolist()
    
    # 如果选择的特征太少，则选择前10个特征
    if len(selected_features) < 5:
        selected_features = xgb_importance.head(10)['Feature'].tolist()
    
    print(f"XGBoost选择的特征 ({len(selected_features)}): {selected_features}")
    
    return selected_features

def lasso_feature_selection(X_train, y_train, feature_names):
    """
    使用L1正则化(LASSO)进行特征选择
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        
    Returns:
        list: 选择的特征列表
        DataFrame: LASSO特征重要性
    """
    print("\n=== LASSO特征选择 ===")
    
    # 确保figures目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    
    # 使用交叉验证找到最佳的alpha值
    print("寻找最佳的LASSO正则化强度(alpha)...")
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train)
    best_alpha = lasso_cv.alpha_
    print(f"最佳alpha值: {best_alpha:.6f}")
    
    # 使用最佳alpha训练LASSO模型
    lasso = Lasso(alpha=best_alpha, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(lasso.coef_)
    })
    feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
    
    # 选择非零系数的特征
    selected_features = feature_importance[feature_importance['Coefficient'] > 0]['Feature'].tolist()
    
    print(f"LASSO选择的特征 ({len(selected_features)}个特征): {selected_features}")
    
    # 可视化LASSO系数
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(15))
    plt.title('Feature Importance (LASSO)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lasso_importance.png'))
    plt.close()
    
    return selected_features, feature_importance

def tree_based_feature_importance(X_train, y_train, feature_names):
    """
    使用基于树的模型计算特征重要性
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        
    Returns:
        dict: 树模型特征重要性结果
    """
    print("\n=== 基于树的特征重要性 ===")
    results = {}
    
    # 随机森林特征重要性
    print("\n随机森林特征重要性:")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })
    rf_importance = rf_importance.sort_values('Importance', ascending=False)
    results['random_forest'] = rf_importance
    
    # 选择重要特征
    rf_threshold = 0.02  # 可以调整此阈值
    rf_selected = rf_importance[rf_importance['Importance'] > rf_threshold]['Feature'].tolist()
    
    print(f"随机森林选择的特征 ({len(rf_selected)}个特征): {rf_selected}")
    
    # 可视化随机森林特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_importance.head(15))
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rf_importance.png'))
    plt.close()
    
    # XGBoost特征重要性
    print("\nXGBoost特征重要性:")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    
    # 获取特征重要性
    xgb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    })
    xgb_importance = xgb_importance.sort_values('Importance', ascending=False)
    results['xgboost'] = xgb_importance
    
    # 选择重要特征
    xgb_threshold = 0.02  # 可以调整此阈值
    xgb_selected = xgb_importance[xgb_importance['Importance'] > xgb_threshold]['Feature'].tolist()
    
    print(f"XGBoost选择的特征 ({len(xgb_selected)}个特征): {xgb_selected}")
    
    # 可视化XGBoost特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(15))
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'xgb_importance.png'))
    plt.close()
    
    # 比较两种树模型的特征重要性排名
    print("\n比较树模型特征重要性排名:")
    
    # 合并两个模型的结果
    compare_df = pd.DataFrame({'Feature': feature_names})
    compare_df = compare_df.merge(
        rf_importance[['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'}),
        on='Feature'
    )
    compare_df = compare_df.merge(
        xgb_importance[['Feature', 'Importance']].rename(columns={'Importance': 'XGB_Importance'}),
        on='Feature'
    )
    
    # 计算特征在每个模型中的排名
    compare_df['RF_Rank'] = compare_df['RF_Importance'].rank(ascending=False)
    compare_df['XGB_Rank'] = compare_df['XGB_Importance'].rank(ascending=False)
    
    # 计算平均排名
    compare_df['Average_Rank'] = (compare_df['RF_Rank'] + compare_df['XGB_Rank']) / 2
    compare_df = compare_df.sort_values('Average_Rank')
    
    results['tree_comparison'] = compare_df
    
    # 选择基于平均排名的特征
    rank_threshold = 10  # 选择平均排名前10的特征
    rank_selected = compare_df.head(rank_threshold)['Feature'].tolist()
    
    print(f"基于平均排名选择的特征 ({len(rank_selected)}个特征): {rank_selected}")
    
    # 可视化特征排名比较
    plt.figure(figsize=(14, 10))
    
    # 散点图：两个模型的特征重要性比较
    plt.scatter(compare_df['RF_Importance'], compare_df['XGB_Importance'], alpha=0.7)
    
    # 添加特征标签
    for i, row in compare_df.iterrows():
        plt.annotate(row['Feature'], 
                    (row['RF_Importance'], row['XGB_Importance']),
                    fontsize=9)
    
    plt.xlabel('Random Forest Importance')
    plt.ylabel('XGBoost Importance')
    plt.title('Feature Importance Comparison: Random Forest vs XGBoost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tree_importance_comparison.png'))
    plt.close()
    
    return results, rf_selected, xgb_selected, rank_selected

if __name__ == "__main__":
    # 这个文件通常从主文件调用，不单独运行
    print("此文件包含嵌入法特征选择功能，应从主程序调用。") 