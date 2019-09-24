import pandas as pd
sub1 = pd.read_csv('../output/auto_lgb.csv')
sub2 = pd.read_csv('../output/NN2.csv')

target = 'target'

final_sub=sub1.copy()
final_sub[target] = final_sub[target]*0.8 + sub2[target]*0.2
final_sub.to_csv('../output/ens1.csv', index=False)