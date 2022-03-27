import os
import numpy as np
import pandas as pd
import addpath
import json
import datetime
from jqdatasdk import *
auth('13350103318', '87654321wW')

pd.set_option('display.max_columns', None) #显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 4000) #页面宽度

def delect_stop(stocks, beginDate, n=30*3):
    stockList = []
    beginDate = datetime.datetime.strptime(beginDate, '%Y-%m-%d')
    for stock in stocks:
        start_date = get_security_info(stock).start_date
        if start_date < (beginDate - datetime.timedelta(days=n)).date():
            stockList.append(stock)
    return stockList

def delect_st(stocks, beginDate):
    st_data = get_extras('is_st', stocks, count=1, end_date=beginDate)
    stockList = [stock for stock in stocks if st_data[stock][0] == False]
    return stockList

if __name__ == '__main__':
    with open(os.path.join(addpath.data_path, 'cn_data', 'factors', 'processed_factor_data_json.json'), 'r') as f:
        data = json.load(f)
    data.keys()
    for date in data.keys():
        data[date] = pd.DataFrame(data[date])
    factor_data_dict = data
    del data

    # 交易时间
    dateList = list(factor_data_dict.keys())
    # 训练集长度
    ts = datetime.datetime.now()
    train_length = 48
    train_data = pd.DataFrame()
    for date in dateList[:train_length]:
        securities_list = delect_stop(get_index_stocks('000300.XSHG', date), date, 90)
        securities_list = delect_st(securities_list, date)
        train_df = factor_data_dict[date]
        data_close = get_price(securities_list, date, dateList[dateList.index(date)+1], '1d', 'close', panel=True)
        data_close.set_index('code', inplace=True)
        train_df['return'] = data_close['close'].groupby(data_close.index).apply(lambda x: x.iloc[-1]/x.iloc[0]-1)
        train_df.dropna(axis=0, subset=['return'], inplace=True)
        train_df = train_df.sort_values(by='return', ascending=False)
        # 选取收益前30%与后30%的数据，并标记label，前面为1，后面为-1
        train_df = train_df.iloc[:int(len(train_df['return'])/10*3), :]\
            .append(train_df.iloc[int(len(train_df['return'])/10*7):, :])
        train_df['label'] = list(train_df['return']\
            .apply(lambda x: 1 if x > np.median(list(train_df['return'])) else 0))
        if train_data.empty:
            train_data = train_df
        else:
            train_data = train_data.append(train_df)
    train_data.to_csv(os.path.join(addpath.data_path, 'cn_data', 'factors', 'train_data1.csv'))
    te1 = datetime.datetime.now()
    print('train data process: ', te1 - ts)

    # test data
    test_data = {}
    for date in dateList[train_length: -1]:
        securities_list = delect_stop(get_index_stocks('000300.XSHG', date), date, 90)
        securities_list = delect_st(securities_list, date)
        test_df = factor_data_dict[date]
        data_close = get_price(securities_list, date, dateList[dateList.index(date) + 1], '1d', 'close', panel=True)
        data_close.set_index('code', inplace=True)
        test_df['return'] = data_close['close'].groupby(data_close.index).apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1)
        test_df.dropna(axis=0, subset=['return'], inplace=True)
        test_df = test_df.sort_values(by='return', ascending=False)
        # 选取收益前30%与后30%的数据，并标记label，前面为1，后面为-1
        test_df = test_df.iloc[:int(len(test_df['return']) / 10 * 3), :] \
            .append(test_df.iloc[int(len(test_df['return']) / 10 * 7):, :])
        test_df['label'] = list(test_df['return'] \
            .apply(lambda x: 1 if x > np.median(list(test_df['return'])) else 0))
        test_data[date] = test_df

    te2 = datetime.datetime.now()
    print('test data process ', te2 - te1)

    test_dict = {}
    for date in test_data.keys():
        test_dict[date] = test_data[date].to_dict()

    jsonObj = json.dumps(test_dict)
    fileObj = open(os.path.join(addpath.data_path, 'cn_data', 'factors', 'test_data_json.json'), 'w',
                   encoding='utf-8')
    fileObj.write(jsonObj)
    fileObj.close()

    print('done')
