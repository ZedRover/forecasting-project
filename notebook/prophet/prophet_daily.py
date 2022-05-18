#
import pandas as pd
from prophet import Prophet
import numpy as np
import os
import sys
sys.path.append('../')
import config
from multiprocessing import Pool
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')
#
df = pd.read_csv('../../data/data_daily_with_aqi.csv')

#
def run_sub(param):
    print('==========={}==========='.format(param))
    target = param[0]
    city = param[1]
    df_tgt = df[df.type==target].drop('type',axis=1)
    df_tgt_ct = df_tgt[['date',city]].rename(columns={'date':'ds',city:'y'})

    m = Prophet()
    m.fit(df_tgt_ct)

    #
    future = m.make_future_dataframe(periods=15)
    future.tail()

    #
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


    #
    fig1 = m.plot(forecast)
    fig1.savefig('./pic/{}_{}_forecast.png'.format(target,city))

    #
    fig2 = m.plot_components(forecast)
    fig2.savefig('./pic/{}_{}_components.png'.format(target,city))

    #
    

    fig3= plot_plotly(m, forecast)
    fig3.write_image('./pic/{}_{}_p_forecast.png'.format(target,city), engine="kaleido")


    #
    fig4 = plot_components_plotly(m, forecast)
    fig4.write_image('./pic/{}_{}_p_components.png'.format(target,city), engine="kaleido")

if __name__ == '__main__':
    
    targets = config.TARGET
    cities = config.CITY
    param = [[i,j] for i in targets for j in cities]
    with Pool(processes=20) as pool:
        pool.map(run_sub, param)
#


#



