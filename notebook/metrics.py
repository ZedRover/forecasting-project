import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
def calc_metric(true,pred,model_name):
    res = {}
    res['MAE'] = mean_absolute_error(true,pred)
    res['MSE'] = mean_squared_error(true,pred)
    res['MAPE'] = mean_absolute_percentage_error(true,pred)
    res['r2'] = r2_score(true,pred)
    return pd.DataFrame(res,index=[model_name])