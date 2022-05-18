#
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool#
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
#

df = pd.read_csv('../../data/data_daily_with_aqi.csv')
df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

def run_sub(param):
    target = param[0] # PM2.5_24h, NO2_24h, O3_24h, CO_24h, SO2_24h, NO2_24h
    city =param[1]
    df_tgt = df[df.type==target].drop('type',axis=1)
    df_tgt_ct = df_tgt[['date',city]].rename(columns={'date':'ds',city:'y'})
    df_tgt_ct = df_tgt_ct.dropna()

    #
    y = df_tgt_ct.resample('M').mean()

    #
    y = y.reset_index(drop=True)

    #
    # Original Series
    fig, axes = plt.subplots(3, 2, figsize=(15,10),sharex=True)
    axes[0, 0].plot(y); axes[0, 0].set_title('Original Series')
    plot_acf(y, ax=axes[0, 1],lags=60)

    # 1st Differencing
    axes[1, 0].plot(y.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(y.diff().dropna(), ax=axes[1, 1],lags=60)

    # 2nd Differencing
    axes[2, 0].plot(y.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(y.diff().diff().dropna(), ax=axes[2, 1],lags=60)

    fig.savefig('./pic/raw/{}_{}_describe.png'.format(target,city))

    #
    

    model = ARIMA(y, order=(1,0,1))
    model_fit = model.fit()
    #
    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig('./pic/raw/{}_{}_residuals.png'.format(target,city))

#


if __name__ == '__main__':
    
    targets = config.TARGET
    cities = config.CITY
    param = [[i,j] for i in targets for j in cities]
    with Pool(processes=20) as pool:
        pool.map(run_sub, param)
