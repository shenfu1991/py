import pandas as pd
import os

def process_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            df = df[df['result'].apply(lambda x: x not in ['SN', 'LN'])]
            new_filename = os.path.splitext(filename)[0] + '_n.csv'
            df.to_csv(os.path.join(folder_path, new_filename), index=False)

process_files('your_folder_path')
