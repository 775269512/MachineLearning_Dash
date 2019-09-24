feature = ""
target = ""
temp = X.groupby([feature])[target].mean().reset_index()
temp.index = temp[feature]
temp = temp.drop(feature, axis=1)
faeture_map = temp.to_dict()[target]

X[feature + "_target_mean"] = X[feature].map(faeture_map)
test_X[feature + "_target_mean"] = test_X[feature].map(faeture_map)