import pandas as pd
import os
import numpy as np

df = pd.read_csv('submission.csv', index_col="id")
print(df.index.name)
print(df.head())

ind_val = np.array(df.index)
ind_eval = [x.replace("validation", "evaluation") for x in ind_val]

print(len(ind_eval))
print(ind_eval)

col = ["id"].extend(np.array(df.columns).tolist())
df_1 = pd.DataFrame(np.zeros((30490,28)), index=ind_eval, columns=df.columns)
df_1.index.name = "id"
print(df_1.index.name)
print(df_1.head())

df.append(df_1)
print(df.tail())
df_1.to_csv(r'submission_fixed.csv')