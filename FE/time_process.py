def get_time_feature(df, col, keep=False):
    """
    为df增加时间特征列,包括:年,月,日,小时,dayofweek,weekofyear
    :param df:
    :param col: 时间列的列名
    :param keep: 是否保留原始时间列
    :return:
    """
    df_copy = df.copy()
    prefix = col + "_"

    df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    df_copy[prefix + 'hour'] = df_copy[col].dt.hour
    df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    # df_copy[prefix + 'holiday'] = df_copy[[prefix + 'year', prefix + 'month', prefix + 'day']] \
    #                                                 .apply(lambda x: _getholiday(x), axis=1)
    if keep:
        return df_copy
    else:
        return df_copy.drop([col], axis=1)