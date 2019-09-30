# shift获得当前value和前后value的delta值
# 距离上一笔以及下一笔交易的时间差特征(seconds), 交易金额差值
key = ['uid']
values = ['TransactionDT', 'TransactionAmt']
for value in values:
    stat_temp = df[key + [value]].copy()
    for i in [-1, 1]:
        shift_value = stat_temp.groupby(key)[value].shift(i)
        cname = '_'.join(key) + '_' + value + '_diff_time{}'.format(i)
        df[cname] = stat_temp[value] - shift_value

# shift获得前后value的值
# 上一笔和下一笔的交易信息
key = ['uid']
values = ['D10', 'D15']
for value in values:
    stat_temp = df[key + [value]].copy()
    for i in [-1, 1]:
        shift_value = stat_temp.groupby(key)[value].shift(i)
        cname = '_'.join(key) + '_' + value + '_shift{}'.format(i)
        df[cname] = shift_value
