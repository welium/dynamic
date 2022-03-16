import numpy as np
import pandas as pd

FileList = pd.read_csv("/home/whngak/fyp3/dynamic/a4c-video-dir/FileList_Semi.csv")

values = []

for i, row in FileList.iterrows():
    temp_row = row
    if row['Split'] == 'TRAIN':
        values.append(temp_row['EF'])

mean = np.mean(values)
std = np.std(values)

print(f"mean is {mean}, std is {std}")

