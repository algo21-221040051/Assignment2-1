import datetime
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


def get_train_test_df():
    with open(os.path.join(addpath.data_path, 'cn_data', 'factors', 'test_data_json.json'), 'r') as f:
        data = json.load(f)
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

    test_dict = dict((key, value.dropna(axis=1)) for key, value in data.items() if key in test_period)

    return train_df, test_dict


def params_select(model_name, train_df):
    '''
    传入模型名称，训练集df
    :return: 输出各参数下模型的评价结果，并保存到本地
    '''
    t1 = datetime.datetime.now()
    res_df = pd.DataFrame()
    score_methods = ['roc_auc', 'f1']
    y_train = train_df['label']
    X_train = train_df.copy()
    try:
        del X_train['return']
    except:
        print('None exists in return')
    del X_train['label']

    if model_name == 'XGBClassifier':
        model = eval('%s(use_label_encoder=False, verbosity=0)' % model_name)
        max_depth = [3, 4, 5, 6, 7, 8]
        sub_samples = [0.6, 0.7, 0.8, 0.9, 1]
        transform = SelectPercentile(f_classif, percentile=80)
        model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])
        cv = StratifiedKFold(5)
        model_list = []
        md_list = []
        ss_list = []
        m1_list = []
        s1_list = []
        m2_list = []
        s2_list = []

        for md in max_depth:
            for ss in sub_samples:
                score_list = []
                print('%s set max_depth: %s, subsample: %s' % (model_name, md, ss))
                for score_method in score_methods:
                    model_pipe.set_params(model__max_depth=md)
                    model_pipe.set_params(model__subsample=ss)
                    score_tmp = cross_val_score(model_pipe, X_train, y_train, scoring=score_method, cv=cv, n_jobs=-1)
                    score_list.append(score_tmp)
                score_df = pd.DataFrame(np.array(score_list), index=score_methods)
                score_mean = score_df.mean(axis=1).rename('mean')
                score_std = score_df.std(axis=1).rename('std')
                score_pd = pd.concat([score_df, score_mean, score_std], axis=1)

                model_list.append(model_name)
                md_list.append(md)
                ss_list.append(ss)
                m1_list.append(score_mean[0])
                s1_list.append(score_std[0])
                m2_list.append(score_mean[1])
                s2_list.append(score_std[1])

                print(score_pd.round(4))
                print('-' * 60)
        res_df = pd.DataFrame({'model':model_list, 'max_depth':md_list, 'subsample':ss_list,
                               '%s_mean' % score_methods[0]: m1_list, '%s_std' % score_methods[0]: s1_list,
                              '%s_mean' % score_methods[1]: m2_list, '%s_std' % score_methods[1]: s2_list})
        res_df.to_csv(os.path.join(addpath.result_path, 'params_selection_%s.csv' % model_name), index=None)
        t2 = datetime.datetime.now()
        print('time: %s' % (t2 - t1))
        return res_df

    elif model_name == 'RandomForestClassifier':
        model = eval('%s()' % model_name)
        n_estimators = [20, 50, 80, 100, 300, 500]
        max_depth = [3, 4, 5, 6, 7, 8]
        transform = SelectPercentile(f_classif, percentile=90)
        model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])
        cv = StratifiedKFold(5)

        model_list = []
        md_list = []
        ss_list = []
        m1_list = []
        s1_list = []
        m2_list = []
        s2_list = []

        for md in max_depth:
            for ne in n_estimators:
                score_list = []
                print('%s set max_depth: %s' % (model_name, md))
                for score_method in score_methods:
                    model_pipe.set_params(model__max_depth=md)
                    model_pipe.set_params(model__n_estimators=ne)
                    score_tmp = cross_val_score(model_pipe, X_train, y_train, scoring=score_method, cv=cv, n_jobs=-1)
                    score_list.append(score_tmp)
                score_df = pd.DataFrame(np.array(score_list), index=score_methods)
                score_mean = score_df.mean(axis=1).rename('mean')
                score_std = score_df.std(axis=1).rename('std')
                score_pd = pd.concat([score_df, score_mean, score_std], axis=1)

                model_list.append(model_name)
                md_list.append(md)
                ss_list.append(ne)
                m1_list.append(score_mean[0])
                s1_list.append(score_std[0])
                m2_list.append(score_mean[1])
                s2_list.append(score_std[1])

                print(score_pd.round(4))
                print('-' * 60)
        res_df = pd.DataFrame({'model': model_list, 'max_depth': md_list, 'n_estimators': ss_list,
                               '%s_mean' % score_methods[0]: m1_list, '%s_std' % score_methods[0]: s1_list,
                               '%s_mean' % score_methods[1]: m2_list, '%s_std' % score_methods[1]: s2_list})
        res_df.to_csv(os.path.join(addpath.result_path, 'params_selection_%s.csv' % model_name), index=None)
        t2 = datetime.datetime.now()
        print('time: %s' % (t2 - t1))
        return res_df


def model_test(train_df, test_dict, model, percentile):
    '''
    传入训练、测试数据，模型，保留特征百分比
    :return: 测试集上各时期的ROC AUC及Accuracy的df
    '''
    y_train = train_df['label']
    X_train = train_df.copy()
    try:
        del X_train['return']
    except:
        print('None exists in return')
    del X_train['label']
    transform = SelectPercentile(f_classif, percentile=percentile)
    X_train_final = transform.fit_transform(X_train, y_train)
    model.fit(X_train_final, y_train)

    test_period = list(test_dict.keys())
    res_df = pd.DataFrame()
    miss_factor_count = 0
    for date in test_period:
        y_test = test_dict[date]['label']
        X_test = test_dict[date].copy()
        del X_test['return']
        del X_test['label']
        try:
            X_test_final = transform.transform(X_test)
        except:
            miss_factor = list(set(X_train.columns) - set(X_test.columns))
            X_test['DP'] = 0
            X_test_final = transform.transform(X_test)
            miss_factor_count += len(miss_factor)
        y_pred_tmp = model.predict(X_test_final)
        y_pred = pd.DataFrame(y_pred_tmp, columns=['label_pred'])
        y_pred_prob = pd.DataFrame(model.predict_proba(X_test_final), columns=['prob1', 'prob2'])
        y_pred.set_index(X_test.index, inplace=True)
        y_pred_prob.set_index(X_test.index, inplace=True)
        pred_df = pd.concat([X_test, y_pred, y_pred_prob], axis=1)

        roc_auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # print('Predict date: ', date)
        # print('AUC: ', roc_auc)
        # print('accuracy: ', accuracy)
        # print('-' * 60)
        res_df.loc[date, 'ROC_AUC'] = roc_auc
        res_df.loc[date, 'Accuracy'] = accuracy
    # print('ROC_AUC mean info: ', np.mean(res_df['ROC_AUC'].mean()))
    # print('Accuracy mean info: ', np.mean(res_df['Accuracy'].mean()))
    return res_df


def factor_importance_plot(train_df, model, percentile):
    '''
    传入训练、测试数据，模型，保留特征百分比
    :return: factor importance plot
    '''
    y_train = train_df['label']
    X_train = train_df.copy()
    try:
        del X_train['return']
    except:
        print('None exists in return')
    del X_train['label']
    transform = SelectPercentile(f_classif, percentile=percentile)
    X_train_final = transform.fit_transform(X_train, y_train)
    model.fit(X_train_final, y_train)
    n_features = X_train.shape[1]
    df = pd.DataFrame({'factor': X_train.columns, 'importance':model.feature_importances_})
    df.sort_values(by='importance', ascending=True, inplace=True)
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    plt.barh(range(n_features), df['importance'], align='center')
    plt.yticks(np.arange(n_features), df['factor'])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


def res_plot(res_df):
    date_list = pd.to_datetime(res_df.index)
    y_auc = res_df['ROC_AUC']
    y_acy = res_df['Accuracy']
    fig = plt.figure(figsize=(12,6), tight_layout=True)
    plt.plot(y_auc, label='ROC_AUC')
    plt.plot(y_acy, label='Accuracy', linestyle='--')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    train_df, test_dict = get_train_test_df()
    train_df.dropna(axis=1, inplace=True)
    # # train_df['label'] = train_df['label'].apply(lambda x: 0 if x == -1 else 1)

    ## 选择参数max_depth: 8, subsample: 1 /(6, 0.8), percentile:80
    # params_select('XGBClassifier', train_df)
    ## 选择参数n_estimators: 500, max_depth: 8, percentile:90
    # params_select('RandomForestClassifier', train_df)

    ## 测试集评价
    # RandomForest
    model_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=0)
    res_rf_df = model_test(train_df, test_dict, model_rf, 100)
    print('RF ROC_AUC mean info: ', np.mean(res_rf_df['ROC_AUC'].mean()))
    print('RF Accuracy mean info: ', np.mean(res_rf_df['Accuracy'].mean()))
    res_plot(res_rf_df)
    factor_importance_plot(train_df, model_rf, 100)

    # XGBoost
    model_xgb = XGBClassifier(max_depth=8, subsample=1, random_state=0, use_label_encoder=False)
    res_xgb_df = model_test(train_df, test_dict, model_xgb, percentile=80)
    print('XGB ROC_AUC mean info: ', np.mean(res_xgb_df['ROC_AUC'].mean()))
    print('XGB Accuracy mean info: ', np.mean(res_xgb_df['Accuracy'].mean()))
    res_plot(res_xgb_df)
    factor_importance_plot(train_df, model_xgb, 100)


    print('done')