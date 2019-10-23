# time_idx必须是递增的
# 时序特征, 当前样本所属类型为c1, 找到 同属于c1类型 且 时间在当前样本之前的 所有样本 的 标签的平均值

time_idx = "hour_day"
his_target_encoding_features = ['app_category', 'app_domain']

for key in tqdm_notebook(his_target_encoding_features):

    temp_train = train[[time_idx] + [key] + [target]]
    temp_test = test[[time_idx] + [key] + [target]]
    temp = temp_train.append(temp_test)
    temp.index = range(len(temp))

    print(key, temp[key].nunique())

    first_flag = True
    for idx in temp.hour_day.unique():
        if idx == temp.hour_day.unique().min():
            continue
        temp_map = temp.loc[temp[time_idx] < idx, [target] + [key]].groupby([key])[target].mean().reset_index()
        temp_map[time_idx] = idx
        if first_flag:
            maps = temp_map
            first_flag = False
        else:
            maps = maps.append(temp_map)
    maps.rename({target: key + "_his_" + target}, axis=1, inplace=True)

    train = pd.merge(train, maps, on=[key, time_idx], how="left")
    test = pd.merge(test, maps, on=[key, time_idx], how="left")