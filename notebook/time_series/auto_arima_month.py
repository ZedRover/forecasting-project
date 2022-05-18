#
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
import config
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
#
df = pd.read_csv('../../data/data_daily_with_aqi.csv')


def run_sub(param):
    target  =param[0]
    city = param[1]
    df_tgt = df[df.type==target].drop('type',axis=1)
    df_tgt_ct = df_tgt[['date',city]].rename(columns={'date':'ds',city:'y'})
    df_tgt_ct = df_tgt_ct.dropna()

    #
    df_tgt_ct.set_index(pd.DatetimeIndex(df_tgt_ct['ds']),inplace=True)

    #
    df_tgt_ct = df_tgt_ct.drop('ds',axis=1).resample('M').mean()

    #

    model = pm.auto_arima(df_tgt_ct.y, start_p=1, start_q=1,
                        information_criterion='aic',
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=10, max_q=10, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=True,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)


    #
    results_summary = model.summary()
    results_as_html = results_summary.tables[1].as_html()
    pd.read_html(results_as_html, header=0, index_col=0)[0].to_csv('./summary/{}_{}_summary.csv'.format(target,city))


    #
    y = df_tgt_ct.y.reset_index(drop=True)
    # Forecast
    n_periods = 15
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(df_tgt_ct.y), len(df_tgt_ct.y)+n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(y[:],color = 'orange',label='observed')
    plt.plot(fc_series, color='darkgreen',label='forecast')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)
    plt.legend()
    plt.title("Final Forecast of ARIMA")
    plt.savefig('./pic/auto/{}_{}_forecast.png'.format(target,city))
    plt.close()
#
if __name__ == '__main__':
    
    targets = config.TARGET
    cities = config.CITY
    param = [[i,j] for i in targets for j in cities]
    with Pool(processes=20) as pool:
        pool.map(run_sub, param)


