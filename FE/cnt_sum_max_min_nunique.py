import pandas as pd
import numpy as np

df = pd.read_excel("../input/sales_transactions.xlsx")
df['order_price_cnt'] = df.groupby(['order'])['ext price'].transform('count')
df['order_price_sum'] = df.groupby(['order'])['ext price'].transform('sum')
df['order_price_max'] = df.groupby(['order'])['ext price'].transform('max')
df['order_price_min'] = df.groupby(['order'])['ext price'].transform('min')
df['order_price_mean'] = df.groupby(['order'])['ext price'].transform('mean')
df['order_price_mean'] = df.groupby(['order'])['ext price'].transform('median')
df['order_price_sum'] = df.groupby(['uid', 'day'])['ext price'].transform('nunique')
