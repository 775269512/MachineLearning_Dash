import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 500)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, space_eval

from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
from time import time
from tqdm import tqdm_notebook

from xgboost import XGBClassifier
import os

from sklearn.model_selection import KFold
from scipy import stats
from sklearn.metrics import roc_curve

import gc
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('./')

# 函数定义
# ### KS
def KsScore(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    fpr,tpr,thresholds= roc_curve(y_true, y_pred)
    ks_score = max(tpr-fpr)
    return ks_score

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'ks', KsScore(labels, preds), True

# feval=evalerror

def merge_inplace(d1, d2, keys):
    d2 = d1[keys].merge(d2, on = keys, how = 'left')
    d1[d2.columns] = d2
    return d1

def merge_inplace_right_index(d1, d2, keys):
    d2 = d1[keys].merge(d2, left_on = keys, right_index=True, how = 'left')
    d1[d2.columns] = d2
    return d1

# 数据读取
input_path = "../problems/problem_1/train/"
os.listdir(input_path)
train_profile = pd.read_csv(input_path + "train_profile.csv")
train_bankStatement = pd.read_csv(input_path + "train_bankStatement.csv")
train_creditBill = pd.read_csv(input_path + "train_creditBill.csv")
train_behaviors = pd.read_csv(input_path + "train_behaviors.csv")
train_label = pd.read_csv(input_path + "train_label.csv")

input_path = "../problems/problem_1/A/"
os.listdir(input_path)
test_profile = pd.read_csv(input_path + "test_profile_A.csv")
test_bankStatement = pd.read_csv(input_path + "test_bankStatement_A.csv")
test_creditBill = pd.read_csv(input_path + "test_creditBill_A.csv")
test_behaviors = pd.read_csv(input_path + "test_behaviors_A.csv")

uid = "用户标识"

# 合并数据
train = pd.read_hdf("./temp/train_profile.hdf", 'w')
test = pd.read_hdf("./temp/test_profile.hdf", 'w')

# wait to generate: state1_相邻两期账单金额差_stat_bucket + state0_相邻两期账单金额差_stat_bucket
tags = [
    "bankStatement_salary0_Amt_stat",
    "bankStatement_salary1_Amt_stat",
    "bankStatement_type0_Amt_stat",
    "bankStatement_type1_Amt_stat",
    "credit_state1_time_stat",
    "credit_state0_time_stat",
    #         "credit_state_time_stat",  # 无效
    "bankStatement_salary0_time_stat",
    "bankStatement_salary1_time_stat",
    "bankStatement_type0_time_stat",
    "bankStatement_type1_time_stat",
    "behavior_type1_cnt",  # 全量的浏览记录- 行为类型-子类型1 的count,
    #         "behavior_type2_cnt",
    "credit_state1_上期账单金额_stat",  # 全量的信用卡账单信息
    "credit_state1_上期还款金额_stat",
    "credit_state1_本期账单余额_stat",
    "credit_state0_上期账单金额_stat",
    "credit_state0_上期还款金额_stat",
    "credit_state0_本期账单余额_stat",
    "credit_银行标识_nunique",
    "credit_state1_上期未还款金额_stat",
    "credit_state0_上期未还款金额_stat",
    "credit_state1_相邻两期账单金额差_stat",
    "credit_state0_相邻两期账单金额差_stat",
    #         "behavior_father_cnt",       # 全量的浏览记录- 行为类型的count, 有了pivot之后重复了
    #         "behavior_son1_cnt",
    #         "behavior_son2_cnt",
    "behavior_行为类型_nunique",  # 全量的浏览记录- 行为类型的nunique
    "behavior_子类型1_nunique",
    "behavior_子类型2_nunique",
    "behavior_行为类型-子类型1_nunique",
    "behavior_行为类型-子类型2_nunique",
    "behaviors_last_father_cnt",  # 最近一段时间浏览记录
    "behaviors_last_type1_cnt",
    #         "behaviors_last_type2_cnt",
    "credit_last_state1_上期账单金额_stat",  # 最近一段时间信用卡账单信息
    "credit_last_state0_上期账单金额_stat",
    "credit_last_state1_上期还款金额_stat",
    "credit_last_state0_上期还款金额_stat",
    "behaviors_dayofyear_stat",  # 浏览记录day_of_year统计信息
    "behaviors_base",  # 浏览记录基础信息
    "behaviors_行为类型_stat",  # 行为类型的统计信息
    "behaviors_行为类型-子类型1labelEncoder_stat",
    "behaviors_行为类型-子类型2labelEncoder_stat",
    "behaviors_last_行为类型_stat",  # 行为类型(最近一段时间)的统计信息
    "behaviors_last_行为类型-子类型1labelEncoder_stat",  # 行为类型-子类型(最近一段时间)的统计信息
    "behaviors_last_行为类型-子类型2labelEncoder_stat",
    "behavior_father_ratio",
    "bank_base",  # 银行流水基础信息
    "creditBill_base",  # 信用卡账单信息
    "num_of_null",  # 缺失副表的数量
    "credit_爆卡and未足额还款",
    "bankStatement_pivot",  # 三张pivot表
    "behaviors_pivot",  ###
    "creditBill_pivot",
    "cnt_stat",  # cnt dayofyear 的均值,方差,最大值
    "creditBill_pivot_last",
    "runxing_money",  # 爆内存了
    "behavior_type1_type2_cnt",  # 子类型1-子类型2的cnt
    "creditBill_add"
]

father_cnt_month = [
    "behavior_father_cnt_month1",
    "behavior_father_cnt_month2",
    "behavior_father_cnt_month3",
    "behavior_father_cnt_month4",
    "behavior_father_cnt_month5",
    "behavior_father_cnt_month6",
    "behavior_father_cnt_month7",
    "behavior_father_cnt_month8",
    "behavior_father_cnt_month9",
    "behavior_father_cnt_month10",
    "behavior_father_cnt_month11",
    "behavior_father_cnt_month12"
]

tags =  tags + father_cnt_month

for tag in tqdm_notebook(tags):
    print(tag)
    train_gen = pd.read_hdf("./temp/train_" + tag + ".hdf")
    test_gen = pd.read_hdf("./temp/test_" + tag + ".hdf")
    assert train_gen.shape[1] == test_gen.shape[1]
    assert list(train_gen.columns) == list(test_gen.columns)
    print(train_gen.shape, test_gen.shape)
    features = []
    for item in train_gen.columns:
        if item == uid:
            features.append(uid)
        else:
            features.append(tag + str(item).replace(",", "").replace("'", ""))
    train_gen.columns = features
    test_gen.columns = features
    if tag not in ["behavior_type1_cnt", "behavior_type2_cnt",
                   "behavior_father_cnt", "behavior_son1_cnt", "behavior_son2_cnt",
                   "behaviors_last_father_cnt", "behaviors_last_type1_cnt",
                   "behaviors_last_type2_cnt", "behaviors_base", "behavior_father_ratio",
                   "bank_base", "creditBill_base", "behavior_all_cnt", "num_of_null",
                   "bankStatement_pivot", "behaviors_pivot", "creditBill_pivot",
                   "month12_divide_month11", "cnt_stat", "creditBill_last_info",
                   "creditBill_pivot_last", "runxing_money", "runxing_shift",
                   "runxing_shift_bill", "runxing_count_1234", "behavior_type1_type2_cnt",
                   "cnt_ratio"] + father_cnt_month:  # uid是其中一列的hdf
        train = merge_inplace_right_index(train, train_gen, ["用户标识"])
        test = merge_inplace_right_index(test, test_gen, ["用户标识"])
    else:
        train = merge_inplace(train, train_gen, ["用户标识"])
        test = merge_inplace(test, test_gen, ["用户标识"])

# 临时增加做实验

# 实验无效,删除gen_feature
# train = train.drop(gen_features, axis = 1)
# test = test.drop(gen_features, axis = 1)

tag = "meta_1"
print(tag)
train_gen = pd.read_hdf("./temp/train_" + tag + ".hdf")
test_gen = pd.read_hdf("./temp/test_" + tag + ".hdf")

# train_gen = pd.read_hdf("../zrx/temp/train_" + tag + ".hdf")
# test_gen = pd.read_hdf("../zrx/temp/test_" + tag + ".hdf")

assert train_gen.shape[1] == test_gen.shape[1]
assert list(train_gen.columns) == list(test_gen.columns)
print(train_gen.shape, test_gen.shape)

gen_features = [x for x in train_gen.columns if x != uid]

print(X.shape, test_X.shape)

# # 合并多个gen_features
# add_tags = [
#  'credit_time_all_账单时间戳_stat',
#  'credit_time_all_上期账单金额_stat',
#  'credit_time_all_上期还款金额_stat',
#  'credit_time_all_本期账单余额_stat',
#  'credit_time_all_信用卡额度_stat',
#  'credit_time0_账单时间戳_stat',
#  'credit_time0_上期账单金额_stat',
#  'credit_time0_上期还款金额_stat',
#  'credit_time0_本期账单余额_stat',
#  'credit_time0_信用卡额度_stat',
#  'credit_time1_账单时间戳_stat',
#  'credit_time1_上期账单金额_stat',
#  'credit_time1_上期还款金额_stat',
#  'credit_time1_本期账单余额_stat',
#  'credit_time1_信用卡额度_stat']

# for tag in add_tags:
#     print(tag)
#     train_gen = pd.read_hdf("./temp/train_" + tag + ".hdf")
#     test_gen = pd.read_hdf("./temp/test_" + tag + ".hdf")

#     assert train_gen.shape[1] == test_gen.shape[1]
#     assert list(train_gen.columns) == list(test_gen.columns)
#     print(train_gen.shape, test_gen.shape)

#     features = []
#     for item in train_gen.columns:
#         if item == uid:
#             features.append(uid)
#         else:
#             features.append(tag + str(item).replace(",", "").replace("'", ""))
#     train_gen.columns = features
#     test_gen.columns = features

#     X = pd.merge(X, train_gen, on = "用户标识", how = "left")
#     test_X = pd.merge(test_X, test_gen, on = "用户标识", how = "left")

# print(X.shape, test_X.shape)


# 准备训练和预测数据
target = "标签"
y = train_label[target]

X = train
test_X = test

# # 删除uid列
# drop_list = ["用户标识"]
# X      = X.drop(drop_list, axis = 1)
# test_X = test_X.drop(drop_list, axis = 1)

# lgb模型
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
         }

import gc
gc.enable()
gc.collect()


# 将当前特征暂存
# features = []
# for i, item in enumerate(X.columns):
#     if item not in features:
#         features.append(item)
#     else:
#         features.append("f_" + str(i))
# X.columns = features
# test_X.columns = features

# # train
# tag = "X"

# result_path = "./temp/" + tag + ".hdf"
# X.to_hdf(result_path, 'w', complib='blosc', complevel=5)

# # test
# tag = "test_X"

# result_path = "./temp/" + tag + ".hdf"
# test_X.to_hdf(result_path, 'w', complib='blosc', complevel=5)

# 删除特征实验
N_round = 10
res = []
del_tags = ["None#", "behavior_type1_cnt", "bankStatement_pivot", "behaviors_pivot", "creditBill_pivot_last"]
for del_tag in tqdm_notebook(del_tags):
    start_time = time()
    KSs = [];
    AUCs = []
    print("#" * 50)
    print('delete tag {}'.format(del_tag))
    used_features = [x for x in X.columns if del_tag not in x]
    print("shape: ", X[used_features].shape, test_X[used_features].shape)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Training on delete tag {} - fold {}'.format(del_tag, fold_n + 1))
        trn_data = lgb.Dataset(X[used_features].iloc[train_index], label=y.iloc[train_index]) # categorical_feature = category
        val_data = lgb.Dataset(X[used_features].iloc[valid_index], label=y.iloc[valid_index]) # categorical_feature = category
        clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data], verbose_eval=0,
                        early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

        val = clf.predict(X[used_features].iloc[valid_index])
        pred = clf.predict(test_X[used_features])

        fpr, tpr, thresholds = roc_curve(y.iloc[valid_index], val)
        ks_score = max(tpr - fpr)
        auc_score = roc_auc_score(y.iloc[valid_index], val)

        KSs.append(ks_score);
        AUCs.append(auc_score)

    print('delete tag {} finished in {}'.format(del_tag, str(datetime.timedelta(seconds=time() - start_time))))
    print('mean KS: {}, mean AUC: {}'.format(np.mean(KSs), np.mean(AUCs)))
    res.append([del_tag, np.mean(KSs), np.mean(AUCs)])

res = pd.DataFrame(res)
res.columns = ['del_tag', "mean_ks", "mean_auc"]
res = res.sort_values(by="mean_ks")
res


# 基于lgb做特征重要性选择并生成meta feature
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

        trn_data = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
        val_data = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])
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


# 只选择top3500个特征
used_features = list(feature_importances.head(3500).index)

print(len(set(used_features)))

X = X.iloc[:, used_features]
test_X = test_X.iloc[:, used_features]


# 运行bagging模型
# sumbit dataframe
lgb_sub = test[["用户标识"]]
lgb_sub[target] = 0

N_round = 10000
KSs = []
AUCs = []
feature_imp_all = pd.DataFrame()
feature_imp_all['feature'] = X.columns

N_MODEL = 3.0
for model_i in tqdm_notebook(range(int(N_MODEL))):

    if N_MODEL != 1.0:
        params['seed'] = model_i + 1123

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        start_time = time()
        print('Training on model {} - fold {}'.format(model_i + 1, fold_n + 1))

        trn_data = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
        val_data = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])
        clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

        feature_imp_all['model_{}-fold_{}'.format(model_i + 1, fold_n + 1)] = clf.feature_importance()
        pred = clf.predict(test_X)
        val = clf.predict(X.iloc[valid_index])

        fpr, tpr, thresholds = roc_curve(y.iloc[valid_index], val)
        ks_score = max(tpr - fpr)
        auc_score = roc_auc_score(y.iloc[valid_index], val)
        print('KS: {}, AUC: {}'.format(ks_score, auc_score))
        KSs.append(ks_score)
        AUCs.append(auc_score)

        lgb_sub[target] = lgb_sub[target] + pred / n_fold / N_MODEL

        print('Model {} - Fold {} finished in {}'.format(model_i + 1, fold_n + 1,
                                                         str(datetime.timedelta(seconds=time() - start_time))))

# 61, [100]	training's auc: 0.819016	valid_1's auc: 0.795242
# valid_1's auc: 0.82817
# 第一折: 0.5080030487804879, 目标: 0.51

# 加上 uid, //100, KS: 0.507527196642777

print(KSs)
print('Mean KS:', np.mean(KSs))
print(AUCs)
print('Mean AUC:', np.mean(AUCs))


feature_imp_all['average'] = feature_imp_all[[x for x in feature_imp_all.columns if x != "feature"]].mean(axis=1)
feature_imp_all = feature_imp_all.sort_values(by = "average", ascending = False)
feature_imp_all.to_csv('feature_imp_all.csv')


# 输出文件
subname = "lgb5_auc8416_ks5268_3bagging"

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
