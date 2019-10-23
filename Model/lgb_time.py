# 切线下验证集: 最后一天的数据作为线下验证
train_idx = train.loc[train["day"] != 30].index
valid_idx = train.loc[train["day"] == 30].index

print(train.shape, test.shape)

# 删除不使用的特征
not_used = [Id, target] + ["device_ip", "device_id"]

used_features = [x for x in test.columns if x not in not_used]
print(used_features)
print(len(used_features))

quick = True
if quick:
    lr = 0.1
    Early_Stopping_Rounds = 150
else:
    lr = 0.006883242363721497
    Early_Stopping_Rounds = 300

N_round = 5000
Verbose_eval = 100

params =  {'num_leaves': 61,  # 当前base 61
           'min_child_weight': 0.03454472573214212,
           'feature_fraction': 0.3797454081646243,
           'bagging_fraction': 0.4181193142567742,
           'min_data_in_leaf': 96,  # 当前base 106
           'objective': 'binary',
           'max_depth': -1,
           'learning_rate': lr,   # 快速验证
    #      'learning_rate': 0.006883242363721497,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "metric": 'binary_logloss', # auc
    #           "metric": 'None',
              "verbosity": -1,
           'reg_alpha': 0.3899927210061127,
           'reg_lambda': 0.6485237330340494,
           'random_state': 47,
    #      'n_jobs': 16,
           'num_threads': 16
    #      'is_unbalance':True
             }

lgb_sub = sub
lgb_sub[target] = 0

category = ['f1', 'f2']

from sklearn.metrics import log_loss

LOSSes = []
AUCs = []
feature_importances = pd.DataFrame()
feature_importances['feature'] = train[used_features].columns
Frac = 0.2

N_MODEL = 1.0
for model_i in tqdm_notebook(range(int(N_MODEL))):

    if N_MODEL != 1.0:
        params['seed'] = model_i + 1123

    start_time = time()
    print('Training on model {}'.format(model_i + 1))

    ## All data
    trn_data = lgb.Dataset(train.iloc[train_idx][used_features], label=train.iloc[train_idx][target],
                           categorical_feature=category)
    val_data = lgb.Dataset(train.iloc[valid_idx][used_features], label=train.iloc[valid_idx][target],
                           categorical_feature=category)

    #     ## sample data
    #     trn_data = lgb.Dataset(train.iloc[train_idx].sample(frac=Frac, random_state=1)[used_features], label=train.iloc[train_idx].sample(frac=Frac, random_state=1)[target], categorical_feature=category)
    #     val_data = lgb.Dataset(train.iloc[valid_idx].sample(frac=Frac, random_state=1)[used_features], label=train.iloc[valid_idx].sample(frac=Frac, random_state=1)[target], categorical_feature=category)

    clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                    verbose_eval=Verbose_eval, early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

    val = clf.predict(train.iloc[valid_idx][used_features])
    loss_ = log_loss(train.iloc[valid_idx][target], val)
    auc_score = roc_auc_score(train.iloc[valid_idx][target], val)

    print('LOSS: {}, AUC: {}'.format(loss_, auc_score))
    LOSSes.append(loss_)
    AUCs.append(auc_score)

    feature_importances['model_{}'.format(model_i + 1)] = clf.feature_importance()

    ## 用数据集重新训练
    print("ReTraining on all data")

    gc.enable()
    del trn_data, val_data
    gc.collect()

    # todo: 不同的迭代倍数尝试, 1.15->1.10

    trn_data = lgb.Dataset(train[used_features], label=train[target], categorical_feature=category)
    clf = lgb.train(params, trn_data, num_boost_round=int(clf.best_iteration * 1.15),
                    valid_sets=[trn_data], verbose_eval=Verbose_eval)  # , feval=evalerror

    pred = clf.predict(test[used_features])
    lgb_sub[target] = lgb_sub[target] + pred / N_MODEL

    print('Model {} finished in {}'.format(model_i + 1, str(datetime.timedelta(seconds=time() - start_time))))