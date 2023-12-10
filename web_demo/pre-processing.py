import pandas as pd

DR_csv_file_path = './Tooth-Final-Project/web_demo/val_gt.csv'
Model_csv_file_path = './Tooth-Final-Project/web_demo/val_prediction.csv'

# sort
df = pd.read_csv(DR_csv_file_path)
df['tooth'] = None 
for row in range(len(df)):
    a = df.iloc[row, 0].split('_')[1]
    b = df.iloc[row, 0].split('_')[0]
    df.iloc[row, 0] = a 
    df.iloc[row, 3] = b
DR_df = df.sort_values(by=['file_name', 'tooth'], ignore_index=True)

# sort
df = pd.read_csv(Model_csv_file_path)
Model_df = df.sort_values(by=['file', 'tooth'], ignore_index=True)

# compare
Model_df['DR'] = None
for row in range(len(Model_df)):
    if Model_df.iloc[row, 0] == DR_df.iloc[row, 0] and str(Model_df.iloc[row, 1]) == DR_df.iloc[row, 3]:
        Model_df.iloc[row, 3] = DR_df.iloc[row, 2]
    else:
        break

# save
Model_df.to_csv('./Tooth-Final-Project/web_demo/val_prediction_final.csv', index=False)