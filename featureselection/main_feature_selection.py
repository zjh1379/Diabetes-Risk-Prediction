#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征选择主程序
这个文件用于集成和运行各种特征选择方法
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 获取当前文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
FIGURES_DIR = os.path.join(CURRENT_DIR, 'figures')
REPORTS_DIR = os.path.join(CURRENT_DIR, 'reports')

# 导入特征选择模块
from .feature_selection import (
    load_and_preprocess_data,
    perform_filter_selection
)

from .wrapper_methods import (
    perform_rfe_selection,
    perform_sequential_selection
)

from .embedded_methods import (
    perform_lasso_selection,
    perform_rf_selection,
    perform_xgboost_selection
)

from .analysis import (
    consolidate_feature_selection,
    stability_analysis,
    learning_curves,
    generate_final_report
)

# 设置随机种子确保结果可重复
RANDOM_STATE = 42

def main():
    """主函数，运行完整的特征选择流程"""
    print("===== 糖尿病风险预测特征选择分析 =====")
    
    # 确保必要的目录存在
    for dir_path in [FIGURES_DIR, REPORTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
    # 1. 数据加载与预处理
    print("\n1. 数据加载与预处理")
    data_path = os.path.join(DATA_DIR, 'diabetes_012_health_indicators_BRFSS2015.csv')
    X, y, feature_names = load_and_preprocess_data(data_path)
    print(f"* 数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"* 特征列表: {feature_names}")
    
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"* 训练集大小: {X_train.shape[0]} 样本")
    print(f"* 测试集大小: {X_test.shape[0]} 样本")
    
    # 2. 运行各种特征选择方法
    print("\n2. 运行特征选择方法")
    
    # 存储各种方法的结果
    all_feature_sets = {}
    
    # 2.1 过滤法特征选择
    print("\n2.1 基于过滤的特征选择")
    filter_selected = perform_filter_selection(X_train, y_train, feature_names)
    all_feature_sets["Filter"] = filter_selected
    
    # 2.2 包装法特征选择
    print("\n2.2 基于包装的特征选择")
    rfe_selected = perform_rfe_selection(X_train, y_train, feature_names)
    all_feature_sets["RFE"] = rfe_selected
    
    seq_selected = perform_sequential_selection(X_train, y_train, feature_names)
    all_feature_sets["Sequential"] = seq_selected
    
    # 2.3 嵌入法特征选择
    print("\n2.3 基于嵌入的特征选择")
    lasso_selected = perform_lasso_selection(X_train, y_train, feature_names)
    all_feature_sets["LASSO"] = lasso_selected
    
    rf_selected = perform_rf_selection(X_train, y_train, feature_names)
    all_feature_sets["RandomForest"] = rf_selected
    
    xgb_selected = perform_xgboost_selection(X_train, y_train, feature_names)
    all_feature_sets["XGBoost"] = xgb_selected
    
    # 3. 综合分析特征选择结果
    print("\n3. 特征选择结果综合分析")
    feature_scores, candidate_subsets = consolidate_feature_selection(all_feature_sets, feature_names)
    
    # 4. 对候选特征子集进行稳定性分析
    print("\n4. 特征稳定性分析")
    stability_results = stability_analysis(X, y, candidate_subsets)
    
    # 5. 学习曲线分析
    print("\n5. 学习曲线分析")
    learning_curves(X, y, candidate_subsets)
    
    # 6. 确定最佳特征子集
    print("\n6. 确定最佳特征子集")
    # 根据AUC选择最佳子集
    best_subset = stability_results.sort_values('Mean_AUC', ascending=False).iloc[0]['Subset']
    best_features = candidate_subsets[best_subset]
    
    print(f"* 最佳特征子集: {best_subset}")
    print(f"* 包含特征 ({len(best_features)}): {best_features}")
    
    # 7. 保存特征比较
    plt.figure(figsize=(14, 8))
    
    # 绘制条形图
    x = np.arange(len(candidate_subsets))
    width = 0.4
    
    # 计算每个子集的特征数量
    subset_feature_counts = [len(features) for subset, features in candidate_subsets.items()]
    
    bars = plt.bar(x, subset_feature_counts, width, label='特征数量')
    
    # 添加高亮显示最佳子集
    best_idx = list(candidate_subsets.keys()).index(best_subset)
    bars[best_idx].set_color('red')
    
    # 设置图表属性
    plt.title('特征子集比较')
    plt.xticks(x, candidate_subsets.keys(), rotation=45)
    plt.xlabel('特征子集')
    plt.ylabel('特征数量')
    
    # 添加文本标注
    for i, v in enumerate(subset_feature_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.axhline(y=X.shape[1], color='gray', linestyle='--', alpha=0.7, label='全部特征')
    plt.text(len(subset_feature_counts)-1, X.shape[1] + 0.5, f'全部特征: {X.shape[1]}', ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_subset_comparison.png'))
    plt.close()
    
    # 8. 生成最终报告
    print("\n7. 生成最终报告")
    generate_final_report(feature_scores, stability_results, best_subset, best_features)
    
    print("\n===== 特征选择分析完成 =====")
    
    # 返回最佳特征集，以便后续模型训练使用
    return best_features

if __name__ == "__main__":
    main() 