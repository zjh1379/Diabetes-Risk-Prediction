#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
糖尿病风险预测项目的特征选择分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可重复
RANDOM_STATE = 42

# 获取当前文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CURRENT_DIR, 'figures')

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
    """
    print(f"加载数据集: {file_path}")
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
        
    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"数据集成功加载，包含 {df.shape[0]} 行和 {df.shape[1]} 列")
        
        # 简单数据预处理
        print("进行数据预处理...")
        
        # 检查并处理缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现 {missing_values.sum()} 个缺失值，进行填充...")
            # 对数值列使用中位数填充，分类列使用众数填充
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # 分离特征和目标变量
        X = df.drop('Diabetes_012', axis=1)
        y = df['Diabetes_012']
        
        feature_names = X.columns.tolist()
        
        print(f"特征集大小: {X.shape}")
        print("目标变量分布:")
        value_counts = y.value_counts(normalize=True) * 100
        for idx, count in value_counts.items():
            print(f"  类别 {idx}: {count:.2f}%")
        
        return X, y, feature_names
        
    except Exception as e:
        print(f"加载或预处理数据时出错: {e}")
        raise

def load_data(file_path):
    """
    加载数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        DataFrame: 加载的数据集
    """
    print(f"加载数据集: {file_path}")
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
        
    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"数据集成功加载，包含 {df.shape[0]} 行和 {df.shape[1]} 列")
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

def exploratory_data_analysis(df):
    """
    进行探索性数据分析
    
    Args:
        df: 数据集
        
    Returns:
        None
    """
    print("\n=== 探索性数据分析 ===")
    
    # 基本信息
    print("\n数据基本信息:")
    print(f"数据集形状: {df.shape}")
    print("\n数据类型:")
    print(df.dtypes)
    
    # 描述性统计
    print("\n描述性统计:")
    print(df.describe())
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.concat([missing_values, missing_percent], axis=1)
    missing_data.columns = ['缺失值数量', '缺失百分比']
    print(missing_data[missing_data['缺失值数量'] > 0])
    
    # 目标变量分布
    print("\n目标变量分布:")
    target_counts = df['Diabetes_012'].value_counts()
    print(target_counts)
    print(f"目标变量百分比: \n{(target_counts / len(df) * 100).round(2)}%")
    
    # 创建图形目录
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 目标变量分布可视化
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Diabetes_012', data=df)
    plt.title('Diabetes Distribution')
    plt.xlabel('Diabetes Status (0: No Diabetes, 1: Prediabetes, 2: Diabetes)')
    plt.ylabel('Count')
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'))
    plt.close()
    
    # 特征分布可视化
    print("\n生成特征分布图...")
    feature_columns = df.columns.drop('Diabetes_012')
    
    for i in range(0, len(feature_columns), 4):
        cols = feature_columns[i:i+4]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for j, col in enumerate(cols):
            if j < len(cols):
                if df[col].nunique() < 10:  # 分类特征
                    sns.countplot(x=col, data=df, ax=axes[j], hue='Diabetes_012', palette='viridis')
                else:  # 连续特征
                    for diabetes_value in df['Diabetes_012'].unique():
                        sns.kdeplot(df[df['Diabetes_012'] == diabetes_value][col], 
                                   ax=axes[j], 
                                   label=f'Diabetes: {diabetes_value}')
                axes[j].set_title(f'Distribution of {col}')
                axes[j].set_xlabel(col)
                axes[j].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'feature_distributions_{i}.png'))
        plt.close()
    
    # 相关性分析
    print("\n计算相关性矩阵...")
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'))
    plt.close()
    
    # 目标变量与特征相关性
    target_correlations = correlation_matrix['Diabetes_012'].drop('Diabetes_012').sort_values(ascending=False)
    print("\n特征与目标变量的相关性:")
    print(target_correlations)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=target_correlations.values, y=target_correlations.index)
    plt.title('Feature Correlation with Diabetes')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'target_correlation.png'))
    plt.close()
    
    return target_correlations

def prepare_data(df):
    """
    准备数据集，分离特征和目标变量，划分训练集和测试集
    
    Args:
        df: 数据集
        
    Returns:
        X_train, X_test, y_train, y_test: 划分后的训练集和测试集
        feature_names: 特征名称列表
    """
    print("\n=== 数据准备 ===")
    # 分离特征和目标变量
    X = df.drop('Diabetes_012', axis=1)
    y = df['Diabetes_012']
    
    feature_names = X.columns.tolist()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    return X_train, X_test, y_train, y_test, feature_names

def perform_filter_selection(X_train, y_train, feature_names, k=10):
    """
    使用过滤法进行特征选择并返回选定的特征
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        k: 选择的特征数量
        
    Returns:
        list: 选定的特征列表
    """
    print("执行过滤法特征选择...")
    
    # 获取所有过滤法的结果
    results = filter_method_selection(X_train, y_train, feature_names)
    
    # 综合不同过滤法的结果
    # 采用投票机制：每个方法选出的前k个特征获得一票
    feature_votes = {feature: 0 for feature in feature_names}
    
    for method, df in results.items():
        top_features = df['Feature'].head(k).tolist()
        for feature in top_features:
            feature_votes[feature] += 1
    
    # 按票数排序
    sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
    
    # 选择获得至少2票的特征
    selected_features = [feature for feature, votes in sorted_features if votes >= 2]
    
    # 如果选择的特征太少，则补充到至少k个特征
    if len(selected_features) < k:
        additional_features = [feature for feature, _ in sorted_features 
                             if feature not in selected_features][:k-len(selected_features)]
        selected_features.extend(additional_features)
    
    print(f"过滤法选择的特征 ({len(selected_features)}): {selected_features}")
    return selected_features

def filter_method_selection(X_train, y_train, feature_names):
    """
    使用过滤法进行特征选择
    
    Args:
        X_train: 训练集特征
        y_train: 训练集目标变量
        feature_names: 特征名称列表
        
    Returns:
        dict: 过滤法特征选择结果
    """
    print("\n=== 过滤法特征选择 ===")
    results = {}
    
    # 皮尔逊相关系数
    print("\n计算皮尔逊相关系数...")
    f_scores = f_classif(X_train, y_train)
    f_scores_df = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': f_scores[0],
        'P-Value': f_scores[1]
    })
    f_scores_df = f_scores_df.sort_values('F-Score', ascending=False)
    results['f_classif'] = f_scores_df
    
    print("F-统计量前10个重要特征:")
    print(f_scores_df.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='F-Score', y='Feature', data=f_scores_df.head(15))
    plt.title('Feature Importance (F-Score)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'f_score_importance.png'))
    plt.close()
    
    # 互信息
    print("\n计算互信息...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Mutual Information': mi_scores
    })
    mi_scores_df = mi_scores_df.sort_values('Mutual Information', ascending=False)
    results['mutual_info'] = mi_scores_df
    
    print("互信息前10个重要特征:")
    print(mi_scores_df.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Mutual Information', y='Feature', data=mi_scores_df.head(15))
    plt.title('Feature Importance (Mutual Information)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'mutual_info_importance.png'))
    plt.close()
    
    # 卡方检验
    print("\n进行卡方检验...")
    # 确保所有值都是非负的
    X_train_chi = X_train.copy()
    for col in X_train_chi.columns:
        if X_train_chi[col].min() < 0:
            X_train_chi[col] = X_train_chi[col] - X_train_chi[col].min()
    
    chi2_scores = chi2(X_train_chi, y_train)
    chi2_scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Chi2-Score': chi2_scores[0],
        'P-Value': chi2_scores[1]
    })
    chi2_scores_df = chi2_scores_df.sort_values('Chi2-Score', ascending=False)
    results['chi2'] = chi2_scores_df
    
    print("卡方检验前10个重要特征:")
    print(chi2_scores_df.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Chi2-Score', y='Feature', data=chi2_scores_df.head(15))
    plt.title('Feature Importance (Chi-Square)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'chi2_importance.png'))
    plt.close()
    
    return results

if __name__ == "__main__":
    # 设置数据文件路径
    data_file = os.path.join(CURRENT_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv", "diabetes_012_health_indicators_BRFSS2015.csv")
    
    # 加载数据
    df = load_data(data_file)
    
    # 探索性数据分析
    target_correlations = exploratory_data_analysis(df)
    
    # 准备数据
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    # 过滤法特征选择
    selected_features = perform_filter_selection(X_train, y_train, feature_names) 