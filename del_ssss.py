import os
import pandas as pd

def remove_similar_rows(df, columns, threshold=0.2):
    to_drop = []
    for i in range(1, len(df)):
        similar = True
        for col in columns:
            if abs(df.loc[i, col] - df.loc[i-1, col]) >= threshold:
                similar = False
                break
        if similar:
            to_drop.append(i)
    return df.drop(to_drop).reset_index(drop=True)

def process_all_csv_in_folder(folder_path, output_folder, columns_to_check):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder_path, filename)
            df = pd.read_csv(full_path)
            filtered_df = remove_similar_rows(df, columns_to_check)
            output_path = os.path.join(output_folder, 'filtered_' + filename)
            filtered_df.to_csv(output_path, index=False)
            print(f"Processed {filename}")

# Columns to consider for similarity
columns_to_check = ['rsi', 'so', 'mfi', 'cci']

# Folder containing the CSV files
input_folder = '/Users/xuanyuan/Documents/5m-big-2'

# Folder to save the filtered CSV files
output_folder = '/Users/xuanyuan/Documents/5m-dum'

process_all_csv_in_folder(input_folder, output_folder, columns_to_check)
