# %%
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import kernel_ridge
from sklearn import svm as svm
import xgboost as xgb
import lightgbm as lgb

from utility import grid_search, varify_on_test
from dataloader import Dataloader

data_loader = Dataloader()
data_loader.load_from_csv()
data_loader.spilt_train_test()

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

# lgbr = lgb.LGBMRegressor(num_leaves=4, min_data_in_leaf=13,
#                          max_bin=55, learning_rate=0.05, n_estimators=1000)
# xgbr = xgb.XGBRegressor(booster='gbtree', gamma=0.001,
#                         max_depth=4, min_child_weight=2, n_estimators=150)
# xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=300, gamma=0.0001)


# param_grid = [{'alpha':[0.01,0.1,1.0,10.0]},
# {'alpha':[0.01,0.1,1.0,10.0]},
# {'alpha':[0.01,0.1,1.0,10.0],'l1_ratio':[0.1,0.3,0.5,0.7,0.9]}]

# %% validation
models = [
    lsr,
    regr,
    enr,
    # svr,  # slow
    # krr,
    # gbr,  # inf
    # rfr,  # slow
    xgbr,
    # xgblr,    # nan
    lgbr
]
model_names = [
    'lsr',
    'regr',
    'enr',
    # 'svr',
    # 'krr',
    # 'gbr',
    # 'rfr',
    'xgbr',
    # 'xgblr',
    'lgbr'
]
for idx, model in enumerate(models):
    train_log_loss, test_log_loss, max_log_loss = varify_on_test(
        model, data_loader)
    print('{0:s}  train_log_loss: {1:f}'.format(
        model_names[idx], train_log_loss))
    print('{0:s}  test_log_loss: {1:f}'.format(
        model_names[idx], test_log_loss))
    # print('max_log_loss: {}'.format(max_log_loss))
    print()

# # %%
# # models = [lsr, regr, enr]
# # model_names = ['lsr', 'regr', 'enr']
# # param_grid = [{'alpha': [0.001, 0.005, 0.01, 0.05]},
# #               {'alpha': [10.0, 15.0, 20.0, 25.0]},
# #               {'alpha': [0.001, 0.05, 0.01, 0.05], 'l1_ratio':[0.2, 0.3, 0.4]}]
#
# models = [xgbr]
# model_names = ['xbgr']
# param_grid = [{
#               'max_depth': [10],
#               'learning_rate': [0.1],
#               'n_estimators': [500],
#               #   'min_child_weight': [0, 2, 5, 10, 20],
#               'max_delta_step': [0],
#               'subsample': [0.9],
#               'colsample_bytree': [0.8],
#               'scale_pos_weight': [0, 0.25, 0.5, 0.75, 1]
#               }]
#
#
# # models = [lgbr]
# # model_names = ['lgbr']
# # param_grid = [{'num_leaves': [3, 4, 5, 6], 'min_data_in_leaf': [3, 6, 9, 12], 'max_bin': [35, 55, 75],
# #                'n_estimators': [500, 700, 900, 1100], 'learning_rate': [0.005]}]
#
# for idx, model in enumerate(models):
#     best_model = grid_search(
#         model, data_loader, param_grid[idx], cv=3, verbose=2, model_name=model_names[idx])
#     train_log_loss, test_log_loss = varify_on_test(best_model, data_loader)
#     print('{0:s}  train_log_loss: {1:f}'.format(
#         model_names[idx], train_log_loss))
#     print('{0:s}  test_log_loss: {1:f}'.format(
#         model_names[idx], test_log_loss))
#     print()
