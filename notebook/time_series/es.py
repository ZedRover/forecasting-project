#
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
import config
from multiprocessing import Pool 

#
df = pd.read_csv('../../data/data_daily_with_aqi.csv')

def run_sub(param):
    target = param[0]
    city = param[1]
    df_tgt = df[df.type==target].drop('type',axis=1)
    df_tgt_ct = df_tgt[['date',city]].rename(columns={'date':'ds',city:'y'})
    df_tgt_ct = df_tgt_ct.dropna()
    df_tgt_ct.set_index(pd.DatetimeIndex(df_tgt_ct['ds']),inplace=True)
    df_tgt_ct = df_tgt_ct.drop('ds',axis=1).resample('M').mean()

    #
    y = df_tgt_ct.y[:-20]
    y_val = df_tgt_ct[-20:]

    #
    # 一次
    fit = SimpleExpSmoothing(y,initialization_method='estimated').fit()

    #
    fit.summary()

    #
    idx = list(range(len(df_tgt_ct)))
    plt.figure(figsize=(12,6))
    plt.plot(pd.Series(fit.fittedfcast,index = idx[:len(fit.fittedfcast)]),label = 'forecast')
    plt.plot(pd.Series(y.values,index = idx[:len(y)]),label='real')
    plt.plot(pd.Series(y_val.values.flatten(),index = idx[-len(y_val):]),label='validation')
    plt.plot(pd.Series(fit.forecast(len(y_val)).values.flatten(),index=idx[-len(y_val):]),label='forecast of validation')
    plt.legend()
    plt.savefig('./pic/es/{}_{}_es_1.png'.format(target,city))
    plt.close()
    #
    fit2 = ExponentialSmoothing(y, trend='add').fit(optimized=True)

    #
    fit2.summary()

    #
    plt.figure(figsize=(12,6))
    plt.plot(pd.Series(fit2.fittedfcast,index = idx[:len(fit.fittedfcast)]),label = 'forecast')
    plt.plot(pd.Series(y.values,index = idx[:len(y)]),label='real')
    plt.plot(pd.Series(y_val.values.flatten(),index = idx[-len(y_val):]),label='validation')
    plt.plot(pd.Series(fit2.forecast(len(y_val)).values.flatten(),index=idx[-len(y_val):]),label='forecast of validation')
    plt.legend()
    plt.savefig('./pic/es/{}_{}_es_2.png'.format(target,city))
    plt.close()

    #
    fit3 = Holt(y, exponential=True, damped_trend=True).fit(optimized=True)

    #
    fit3.summary()

    #
    plt.figure(figsize=(12,6))
    plt.plot(pd.Series(fit3.fittedfcast,index = idx[:len(fit3.fittedfcast)]),label = 'forecast')
    plt.plot(pd.Series(y.values,index = idx[:len(y)]),label='real')
    plt.plot(pd.Series(y_val.values.flatten(),index = idx[-len(y_val):]),label='validation')
    plt.plot(pd.Series(fit3.forecast(len(y_val)).values.flatten(),index=idx[-len(y_val):]),label='forecast of validation')
    plt.legend()
    plt.savefig('./pic/es/{}_{}_es_3.png'.format(target,city))
    plt.close()

#



if __name__ == '__main__':
    
    targets = config.TARGET
    cities = config.CITY
    param = [[i,j] for i in targets for j in cities]
    with Pool(processes=20) as pool:
        pool.map(run_sub, param)