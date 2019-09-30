params = {'num_leaves': 41,  # 当前base 61
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 96,  # 当前base 106
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,   # 快速验证
#           'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
#           "metric": 'None',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
#           'n_jobs': 16,
          'num_threads': 16
#           'is_unbalance':True
         }

lr = 0.1
Early_Stopping_Rounds = 100


category = ["C" + str(i) for i in range(1, 27)]
trn_data = lgb.Dataset(X, label=y, categorical_feature = category)

N_round = 30
clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data], verbose_eval=100,
                early_stopping_rounds=Early_Stopping_Rounds)

train_lgb_feature= pd.DataFrame(clf.predict(X, pred_leaf=True))
test_lgb_feature= pd.DataFrame(clf.predict(test_X, pred_leaf=True))

tree_feas = ["lgb_" + str(i) for i in range(1, N_round + 1)]
train_lgb_feature.columns = tree_feas
test_lgb_feature.columns = tree_feas

# 注意: 生成的特征"lgb_x"是类别型变量

train_lgb_feature.shape, test_lgb_feature.shape

train_lgb_feature['lgb_1'].nunique()