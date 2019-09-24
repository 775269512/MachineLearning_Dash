from sklearn import preprocessing
lbl = LabelEncoder()
temp = pd.DataFrame(train[f].astype(str).append(test[f].astype(str)))
lbl.fit(temp[f])
train[f] = lbl.transform(list(train[f].astype(str)))
test[f] = lbl.transform(list(test[f].astype(str)))