import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

confirmed_global = pd.read_csv('../data/time_series_covid19_confirmed_global.csv')
recovered_global = pd.read_csv('../data/time_series_covid19_recovered_global.csv')
deaths_global = pd.read_csv('../data/time_series_covid19_deaths_global.csv')
confirmed = confirmed_global[confirmed_global['Country/Region'] == 'Italy'].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()
recovered = recovered_global[recovered_global['Country/Region'] == 'Italy'].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()
deaths = deaths_global[deaths_global['Country/Region'] == 'Italy'].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()
def run(data, parameter, title):
    indexedDataset = data['10/10/20':'11/10/20']
    indexedDataset_logScale = np.log(indexedDataset) 
    diff1 = indexedDataset_logScale.diff(1)
    diff1.dropna(inplace=True)
    diff2 = diff1.diff(1)
    diff2.dropna(inplace=True)
    try:
        model = ARIMA(indexedDataset_logScale, order=parameter)
        results_ARIMA = model.fit(disp=-1)
    except (ValueError,np.linalg.LinAlgError):
        return
    # plt.plot(diff1)
    # plt.plot(results_ARIMA.fittedvalues, color='red')#模型数据的差分值
    # plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - diff1)**2))

    # predictions_ARIMA_diff_cumsum = results_ARIMA.fittedvalues.cumsum()
    # predictions_ARIMA = indexedDataset_logScale[0]+predictions_ARIMA_diff_cumsum
    # predictions_ARIMA = np.exp(predictions_ARIMA)
    # fig = plt.figure(figsize=(12,8))
    # plt.plot(indexedDataset,label='train')
    # plt.plot(predictions_ARIMA,label='test')
    # plt.xticks(rotation=45)

    predictions_ARIMA_diff_cumsum = results_ARIMA.predict('11/11/20','12/10/20', dynamic=True).cumsum()
    predictions_ARIMA = indexedDataset_logScale[-1]+predictions_ARIMA_diff_cumsum
    predictions_ARIMA = np.exp(predictions_ARIMA)
    fig = plt.figure(figsize=(12,8))
    data = data['10/10/20':'12/10/20']
    data = data.reindex(pd.to_datetime(data.keys()))
    plt.plot(data,label='train')
    plt.plot(predictions_ARIMA,label='test')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    # plt.show()

    MAE = mean_absolute_error(data['11/11/20':'12/10/20'],predictions_ARIMA)
    RMSE = np.sqrt(mean_squared_error(data['11/11/20':'12/10/20'],predictions_ARIMA))
    R2 = r2_score(data['11/11/20':'12/10/20'],predictions_ARIMA)
    print("MAE:{:.4f}, RMSE:{:.4f}, R2:{:.4f}".format(MAE, RMSE, R2))

for i in range(1,9):
    for j in range(1,9):
        print(i,j)
        run(recovered, (i,1,j), 'Denmark_recovered')
# run(confirmed, (1,1,4), 'Italy_confirmed')
# run(recovered, (2,1,4), 'Italy_recovered')
# run(deaths, (1,1,3), 'Italy_deaths')