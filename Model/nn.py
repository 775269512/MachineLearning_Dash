# 定义categorical列
categorical = ["sex", "occupation", "edu", "marry", "register"]
for item in categorical:
    train[item] = train[item].apply(lambda x: str(x))
    test[item] = test[item].apply(lambda x: str(x))

# 定义numerical列
numerical = [c for c in train.columns if c not in categorical + ['label']]

# 数值型特征处理: fillna, 归一化
train = train.fillna(train.mean())
test = test.fillna(test.mean())

drop_cols = []
for column in tqdm_notebook(numerical):
    scaler = StandardScaler()
    if train[column].max() > 100 and train[column].min() >= 0:
        train[column] = np.log1p(train[column])
        test[column] = np.log1p(test[column])
    scaler.fit(np.concatenate([train[column].values.reshape(-1,1), test[column].values.reshape(-1,1)]))
    train[column] = scaler.transform(train[column].values.reshape(-1,1))
    test[column] = scaler.transform(test[column].values.reshape(-1,1))

# 类别型特征缺失值处理和Label Encoding
category_counts = {}
for f in categorical:
    train[f] = train[f].replace("nan", "other")
    train[f] = train[f].replace(np.nan, "other")
    test[f] = test[f].replace("nan", "other")
    test[f] = test[f].replace(np.nan, "other")
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))
    category_counts[f] = len(list(lbl.classes_)) + 1


# 数据输入的准备

# 生成一个字典
def get_input_features(df):
    X = {'numerical':np.array(df[numerical])}
    for cat in categorical:
        X[cat] = np.array(df[cat])
    return X

# 模型定义
def make_model():
    k.clear_session()

    categorical_inputs = []
    for cat in categorical:
        categorical_inputs.append(Input(shape=[1], name=cat))

    categorical_embeddings = []
    for i, cat in enumerate(categorical):
        categorical_embeddings.append(
            Embedding(category_counts[cat], int(np.log1p(category_counts[cat]) + 1), name=cat + "_embed")(
                categorical_inputs[i]))

    categorical_logits = Concatenate(name="categorical_conc")(
        [Flatten()(SpatialDropout1D(.1)(cat_emb)) for cat_emb in categorical_embeddings])
    categorical_logits = Dropout(.25)(categorical_logits)

    numerical_inputs = Input(shape=[train[numerical].shape[1]], name='numerical')
    numerical_logits = Dropout(0)(numerical_inputs)

    x = Concatenate()([
        categorical_logits,
        numerical_logits,
    ])
    #     x = categorical_logits
    x = Dense(512, activation='relu', init='he_normal')(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu', init='he_normal')(x)
    x = BatchNormalization()(x)
    #     x = Dropout(.2)(x)
    x = Dense(64, activation='relu', init='he_normal')(x)
    x = BatchNormalization()(x)
    #     x = Dropout(.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=categorical_inputs + [numerical_inputs], outputs=out)
    loss = "binary_crossentropy"
    model.compile(optimizer=Adagrad(lr=0.001), loss=loss, metrics=['accuracy', auroc])
    return model

target = '标签'
# sumbit dataframe
nn_sub = test[["uid"]]
nn_sub[target] = 0

# 模型训练
nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=1126)
KSs = []

nround = 100
Patience = 7
model_name = "nn2.h5"

X_test = get_input_features(test)

fold_i = 1
for train_idx, valid_idx in skf.split(train, train[target]):

    print("\nfold {}".format(fold_i))
    X_train, y_train = train.iloc[train_idx][features], train.iloc[train_idx][target]
    X_valid, y_valid = train.iloc[valid_idx][features], train.iloc[valid_idx][target]

    X_train = get_input_features(X_train)
    X_valid = get_input_features(X_valid)

    model = make_model()
    best_score = 0
    patience = 0
    for i in range(nround):
        if patience < Patience:
            hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=128, epochs=1, verbose=1)
            valid_preds = model.predict(X_valid, batch_size=800, verbose=True)
            score = roc_auc_score(y_valid, valid_preds)
            print('fold: {}, round: {}, auc: {}'.format(fold_i, i, score))
            if score > best_score:
                model.save_weights(model_name)
                best_score = score
                patience = 0
            else:
                patience += 1

    # 加载最优模型
    model.load_weights(model_name)
    valid_preds = model.predict(X_valid, batch_size=8000, verbose=True)

    fpr, tpr, thresholds = roc_curve(y_valid, valid_preds)
    ks_score = max(tpr - fpr)
    print('KS: {}'.format(ks_score))
    KSs.append(ks_score)

    if fold_i == 1:
        predictions = model.predict(X_test) / nfold
    else:
        predictions += model.predict(X_test) / nfold

    fold_i += 1


print(KSs)
print('Mean KS:', np.mean(KSs))


nn_sub[target] = predictions