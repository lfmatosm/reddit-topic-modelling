#Merges two CSV files and saves the final result

import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("merged.csv")
