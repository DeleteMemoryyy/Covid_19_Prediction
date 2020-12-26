import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize


class SIRModel:

    def __init__(self, beta, gamma, method):
        self.__beta = beta
        self.__gamma = gamma
        self.__method = method
        self.__optimal = None
        self.__predict_loss = None

    def sir_model(self, y0, t, beta, gamma):
        S, I, R = y0
        dSdt = -beta * S * I / (S + I + R)
        dIdt = beta * S * I / (S + I + R) - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def loss_function(self, params, infected, recovered, y0):
        size = len(infected)
        t = np.linspace(1, size, size)
        beta, gamma = params
        solution = odeint(self.sir_model, y0, t, args=(beta, gamma))
        l1 = np.mean((solution[:, 1] - infected) ** 2)
        l2 = np.mean((solution[:, 2] - recovered) ** 2)
        return l1 + l2

    def fit(self, y0, infected, recovered):
        self.__optimal = minimize(self.loss_function, [self.__beta, self.__gamma],
                                  args=(infected, recovered, y0),
                                  method=self.__method,
                                  bounds=[(0.00000001, 1), (0.00000001, 1)])

    def predict(self, test_y0, days):
        predict_result = odeint(self.sir_model, test_y0, np.linspace(1, days, days), args=tuple(self.__optimal.x))
        return predict_result

    def get_optimal_params(self):
        return self.__optimal.x

    def get_predict_loss(self):
        return self.__predict_loss


# get the initial number of susceptible, infectious and recovered people
def get_init_data(N, I0, R0):
    S0 = N - I0 - R0
    return [S0, I0, R0]


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # set params
    country = 'Italy'  # country
    N = 60000000  # total population
    train_start = '10/30/20'
    train_end = '11/10/20'
    valid_start = '11/11/20'
    valid_end = '12/4/20'
    pre_num = 24

    # read input data
    confirmed_global = pd.read_csv('../data/time_series_covid19_confirmed_global.csv')
    recovered_global = pd.read_csv('../data/time_series_covid19_recovered_global.csv')
    deaths_global = pd.read_csv('../data/time_series_covid19_deaths_global.csv')

    confirmed = confirmed_global[confirmed_global['Country/Region'] == country].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()
    recovered = recovered_global[recovered_global['Country/Region'] == country].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()
    deaths = deaths_global[deaths_global['Country/Region'] == country].drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).sum()

    infectious = confirmed - recovered - deaths
    recovered = recovered + deaths

    # training set
    infectious_train = infectious.loc[train_start:train_end]
    recovered_train = recovered.loc[train_start:train_end]
    # print(infectious_train)
    # print(recovered_train)

    # validation set
    infectious_valid = infectious.loc[valid_start:valid_end]
    recovered_valid = recovered.loc[valid_start:valid_end]

    # get initial data
    I0 = infectious_train.loc[train_start]
    R0 = recovered_train.loc[train_start]
    Y0 = get_init_data(N, I0, R0)

    # train
    model = SIRModel(0.0001, 0.0001, 'L-BFGS-B')
    model.fit(Y0, infectious_train, recovered_train)
    best_params = model.get_optimal_params()
    # best_params = model.get_optimal_params()
    # print(best_params)

    # predict
    i0 = infectious_train.loc[train_end]
    r0 = recovered_train.loc[train_end]
    y0 = get_init_data(N, i0, r0)
    predict_result = model.predict(y0, pre_num)
    # print(predict_result)

    t = np.linspace(1, len(infectious_valid), len(infectious_valid))
    t_predict = np.linspace(1, pre_num, pre_num)

    fig = plt.figure(facecolor='w', dpi=100)
    ax = fig.add_subplot(111)

    # real I and R
    ax.plot(t, infectious_valid, 'r', alpha=0.5, lw=2, label='infectious_real')
    ax.plot(t, recovered_valid, 'g', alpha=0.5, lw=2, label='recovered_real')
    # predicted I and R
    ax.plot(t_predict, predict_result[:, 1], 'r-.', alpha=0.5, lw=2, label='infectious_predict')
    ax.plot(t_predict, predict_result[:, 2], 'g-.', alpha=0.5, lw=2, label='recovered_predict')

    ax.set_xlabel('Time/days')
    ax.set_ylabel('Number')

    legend = ax.legend()
    ax.grid(axis='y')
    plt.box(False)
    plt.show()
