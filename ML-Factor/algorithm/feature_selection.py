import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import addpath
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    with open(os.path.join(addpath.data_path, 'cn_data', 'factors', 'test_data_json.json'), 'r') as f:
        data = json.load(f)
    data.keys()
    for date in data.keys():
        data[date] = pd.DataFrame(data[date])
    date_list = list(data.keys())
    train_period = date_list[0: 108]
    test_period = date_list[109:-1]

    train_df = pd.DataFrame()
    for date in train_period:
        if train_df.empty:
            train_df = data[date]
        else:
            train_df = train_df.append(data[date])
    train_df.dropna(axis=1, inplace=True)
    y_train = train_df['label']
    X_train = train_df.copy()
    try:
        del X_train['return']
    except:
        print('None exists in return')
    del X_train['label']

    # SelectPercentile: 使用f_classif时，选取原数据score最高的百分比特征保留下来，
    # score的计算默认基于ANOVA，我的理解是通过方差分析选取差异化最大的一些特征
    transform = SelectPercentile(f_classif)

    def feature_select(model_name):
        model = model_name
        # Pipline先处理SelectPercentile，再建立model
        model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])
        score_means = []
        score_stds = []
        percentiles = (10, 20, 30, 40, 50, 60, 70, 80, 90)
        for percentile in percentiles:
            model_pipe.set_params(ANOVA__percentile=percentile)
            this_scores = cross_val_score(model_pipe, X_train, y_train, cv=5, n_jobs=-1)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())
        plt.errorbar(percentiles, score_means, np.array(score_stds))
        plt.title('Performance of the model-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')
        plt.axis('tight')
        plt.show()
    feature_select(RandomForestClassifier())
    feature_select(XGBClassifier())
    print('done')