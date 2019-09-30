import argparse

ap = argparse.ArgumentParser(description='lgb.py')
ap.add_argument('size', nargs=1, action="store", default=-1, type=int)
ap.add_argument('feature', default=1, nargs=1, action="store",  type=int)

pa = ap.parse_args()
size = pa.size[0]
feature_engineer = pa.feature[0]
if feature_engineer == 1:
    feature_engineer = True
else:
    feature_engineer = False

# 如果设置为-1,使用全量数据,否则使用size大小的数据
if size == -1:
    NROWS = None
else:
    NROWS = size
print("NROWS: ", NROWS)