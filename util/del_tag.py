N_round = 1
res = []
for del_tag in tqdm_notebook(["None#", "behavior_type1_cnt", "bankStatement_pivot", "behaviors_pivot", "creditBill_pivot_last"]):
    start_time = time()
    KSs = [];
    AUCs = []
    print("#" * 50)
    print('delete tag {}'.format(del_tag))
    used_features = [x for x in X.columns if del_tag not in x]
    print("shape: ", X[used_features].shape, test_X[used_features].shape)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Training on delete tag {} - fold {}'.format(del_tag, fold_n + 1))
        trn_data = lgb.Dataset(X[used_features].iloc[train_index], label=y.iloc[train_index])
        val_data = lgb.Dataset(X[used_features].iloc[valid_index], label=y.iloc[valid_index])
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