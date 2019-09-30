n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state = 889)

quick = False
if quick:
    lr = 0.1
    Early_Stopping_Rounds = 150
else:
    lr = 0.006883242363721497
    Early_Stopping_Rounds = 300

params = {'num_leaves': 41,  # 当前base 61
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 96,  # 当前base 106
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': lr,   # 快速验证
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

print(X.shape, test_X.shape)

N_round = 1000
KSs = []
AUCs = []
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

Meta = False
if Meta:
    # save meta feature
    meta_train = pd.DataFrame(X[uid])
    meta_test = pd.DataFrame(test_X[uid])
    meta_train["meta_feature"] = 0
    meta_test["meta_feature"] = 0

N_MODEL = 1.0
for model_i in tqdm_notebook(range(int(N_MODEL))):

    if N_MODEL != 1.0:
        params['seed'] = model_i + 1123

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        start_time = time()
        print('Training on model {} - fold {}'.format(model_i + 1, fold_n + 1))

        trn_data = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index]) # categorical_feature = category
        val_data = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index]) # categorical_feature = category
        clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

        feature_importances['model_{}-fold_{}'.format(model_i + 1, fold_n + 1)] = clf.feature_importance()

        val = clf.predict(X.iloc[valid_index])
        pred = clf.predict(test_X)

        if Meta:
            # meta feature
            meta_train.loc[valid_index, "meta_feature"] = val
            meta_test["meta_feature"] += pred / float(n_fold)

        fpr, tpr, thresholds = roc_curve(y.iloc[valid_index], val)
        ks_score = max(tpr - fpr)
        auc_score = roc_auc_score(y.iloc[valid_index], val)
        print('KS: {}, AUC: {}'.format(ks_score, auc_score))
        KSs.append(ks_score)
        AUCs.append(auc_score)
        print('Model {} - Fold {} finished in {}'.format(model_i + 1, fold_n + 1,
                                                         str(datetime.timedelta(seconds=time() - start_time))))

if Meta:
    # train
    tag = "train_meta_1"
    result_path = "./temp/" + tag + ".hdf"
    meta_train.to_hdf(result_path, 'w', complib='blosc', complevel=5)

    # test
    tag = "test_meta_1"
    result_path = "./temp/" + tag + ".hdf"
    meta_test.to_hdf(result_path, 'w', complib='blosc', complevel=5)

feature_importances['average'] = feature_importances[[x for x in feature_importances.columns if x != "feature"]].mean(axis=1)
feature_importances = feature_importances.sort_values(by = "average", ascending = False)
feature_importances.to_csv('feature_importances.csv')
feature_importances


print(KSs)
print('Mean KS:', np.mean(KSs))
print(AUCs)
print('Mean AUC:', np.mean(AUCs))


subname = "lgb5_auc8425_ks5241_3bagging"

# 处理成0-1之间
max_ = lgb_sub["标签"].max()
min_ = lgb_sub["标签"].min()
lgb_sub["标签"] = lgb_sub["标签"].apply(lambda x: (x - min_) / (max_ - min_) )

lgb_sub.to_csv("./output/" + subname + ".csv", index=False, header = False)
print('-' * 30)
print('Training has finished.')
# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('KSs:', KSs)
print('Mean KS:', np.mean(KSs))
print('-' * 30)