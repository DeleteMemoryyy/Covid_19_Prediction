# %%
import warnings

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import kernel_ridge
from sklearn import linear_model as lm
from sklearn import svm as svm

from dataloader import Dataloader
from utility import Stacking, neg_log_loss

warnings.filterwarnings('ignore')

data_loader = Dataloader()
data_loader.load_from_csv()
data_loader.spilt_train_test()

enable_stacking = True
enable_add_result = True
nn_result_file = ''

# %%
lsr = lm.Lasso(alpha=0.0005547)
regr = lm.Ridge(alpha=15.0)
enr = lm.ElasticNet(alpha=0.0009649, l1_ratio=0.2)
svr = svm.SVR(C=200, gamma=0.001)
krr = kernel_ridge.KernelRidge(kernel='polynomial')
gbr = ensemble.GradientBoostingRegressor(
    loss='huber', max_features='sqrt', n_estimators=400)
rfr = ensemble.RandomForestRegressor(n_estimators=90)
xgbr = xgb.XGBRegressor(booster='gbtree',
                        # max_dept=10,
                        learning_rate=0.1,
                        n_estimators=500,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        scale_pos_weight=1)
xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=300)
lgbr = lgb.LGBMRegressor(num_leaves=6, min_data_in_leaf=12,
                         max_bin=35, learning_rate=0.05, n_estimators=1100)

# %%
models = [
    lsr,
    regr,
    enr,
    # svr,  # slow
    # krr,
    # gbr,  # inf
    # rfr,  # slow
    xgbr,  # nan
    # xgblr,
    lgbr
]
# stacker = [lm.ElasticNet(alpha=0.45, l1_ratio=0.5)]
stacker = [lm.Lasso(alpha=0.01)]
added_result = []
weights = [1.0]
# weights = [0.5, 0.5]

# if enable_add_result:
#     nn_result = pd.read_csv('result/{}'.format(nn_result_file))
#     nn_data = np.transpose(nn_result.values).reshape((6, -1, 60))
#     added_result.append(nn_data.tolist())


stacking = Stacking(5, data_loader.rand_seed, models,
                    stacker, added_result, weights)
# stacking.fit(data_loader.x_train, data_loader.y_train)
# stacking_y_test_predict = stacking.predict(data_loader.x_test)
while True:
    stacking_y_test_predict = stacking.fit_predict(data_loader.x_train, data_loader.y_train,
                                               data_loader.x_test)
y_test_predict = np.array(stacking_y_test_predict).reshape(-1, )
test_log_loss = neg_log_loss(data_loader.y_test, y_test_predict)
print('test_log_loss: {0:f}'.format(test_log_loss))
