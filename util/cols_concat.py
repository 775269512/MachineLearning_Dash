import pandas as pd
import numpy as np

df = pd.read_excel("../input/sales_transactions.xlsx")

def cols_concat(df, con_list):
    name = "_".join(con_list)
    df[name] = df[con_list[0]].astype(str)
    for item in con_list[1:]:
        df[name] = df[name] + '_' + df[item].astype(str)
    return df

df = cols_concat(df, ["f1", "f2"])


