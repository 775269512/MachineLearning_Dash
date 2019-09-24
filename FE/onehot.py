temp = pd.get_dummies(train[item], prefix=item)
train = pd.concat([train, temp], axis = 1)
train = train.drop(item, axis = 1)