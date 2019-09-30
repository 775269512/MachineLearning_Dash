def merge_inplace(d1, d2, keys):
    d2 = d1[keys].merge(d2, on = keys, how = 'left')
    d1[d2.columns] = d2
    return d1

def merge_inplace_right_index(d1, d2, keys):
    d2 = d1[keys].merge(d2, left_on = keys, right_index=True, how = 'left')
    d1[d2.columns] = d2
    return d1