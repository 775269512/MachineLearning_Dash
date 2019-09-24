from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

def get_dc_feature(df_train, df_test, n_comp=12, id_column=None, label_column=None):
    """
    构造分解特征
    """
    train = df_train.copy()
    test = df_test.copy()

    if id_column:
        train_id = train[id_column]
        test_id = test[id_column]
        train = drop_columns(train, [id_column])
        test = drop_columns(test, [id_column])
    if label_column:
        train_y = train[label_column]
        train = drop_columns(train, [label_column])

    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results_train = tsvd.fit_transform(train)
    tsvd_results_test = tsvd.transform(test)

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train)
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=420)
    ica2_results_train = ica.fit_transform(train)
    ica2_results_test = ica.transform(test)

    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results_train = grp.fit_transform(train)
    grp_results_test = grp.transform(test)

    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp_results_train = srp.fit_transform(train)
    srp_results_test = srp.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]

        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]

        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]

    if id_column:
        train[id_column] = train_id
        test[id_column] = test_id
    if label_column:
        train[label_column] = train_y

    return train, test

def drop_columns(df, columns):
    """
    按列删除df的columns列
    :param df:
    :param columns: ["col1", "col2", ...]
    :return:
    """
    df_copy = df.copy()
    df_copy = df_copy.drop(columns, axis=1)
    return df_copy