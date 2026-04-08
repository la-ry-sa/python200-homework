import pandas as pd
import numpy as np

filename_list = []
pattern = "world_happiness/world_happiness_20{}.csv"
for i in range(15, 25):
    filename = pattern.format(i)
    filename_list.append(filename)

print(filename_list)

df_list = []

for file in filename_list:
    df = pd.read_csv(file, on_bad_lines='skip')
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
print(combined_df.head())