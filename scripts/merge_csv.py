#Merges two CSV files and saves the final result

import pandas as pd

df1 = pd.read_csv("etm_results_1.csv")
df2 = pd.read_csv("etm_results.csv")
# df3 = pd.read_csv("ctm_combined_results3.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("etm_full_results.csv")