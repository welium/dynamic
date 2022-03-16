import pandas as pd
import random

orig = pd.read_csv("/home/whngak/fyp3/dynamic/a4c-video-dir/FileList.csv")
semi_csv = pd.DataFrame(columns=orig.columns)

# 1/8 Labelled within train
# 7/8 Unlabelled within train
# Train:Test:Val = 6:2:2


for i, row in orig.iterrows():
    temp_row = row
    temp_rand = random.random()
    if temp_rand < 0.2:
        temp_row['Split'] = 'VAL'
    elif temp_rand < 0.4:
        temp_row['Split'] = 'TEST'
    elif temp_rand < 0.475:
        temp_row['Split'] = 'TRAIN'
    else:
        temp_row['Split'] = 'TRAIN_UNLABELLED'

    semi_csv = semi_csv.append(temp_row)

semi_csv.to_csv('/home/whngak/fyp3/dynamic/a4c-video-dir/FileList_Semi.csv', index=False)