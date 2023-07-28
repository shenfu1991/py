import os
import pandas as pd

def split_each_csv_file(directory):
    # Walk through directory including subdirectories
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                # Load CSV file
                file_path = os.path.join(dirname, filename)
                data = pd.read_csv(file_path)
                
                # Calculate split indices
                total_rows = len(data)
                part_size = total_rows // 3
                first_split = part_size
                second_split = 2 * part_size
                
                # Split DataFrame into three parts
                part_1 = data.iloc[:first_split]
                part_2 = data.iloc[first_split:second_split]
                part_3 = data.iloc[second_split:]
                
                # Generate output file names
                base_name = os.path.splitext(filename)[0]
                output_file_1 = f"{base_name}_part_1.csv"
                output_file_2 = f"{base_name}_part_2.csv"
                output_file_3 = f"{base_name}_part_3.csv"
                
                # Save parts to new CSV files
                part_1.to_csv(os.path.join(dirname, output_file_1), index=False)
                part_2.to_csv(os.path.join(dirname, output_file_2), index=False)
                part_3.to_csv(os.path.join(dirname, output_file_3), index=False)

# Call the function with the directory path


path = "/Users/xuanyuan/Downloads/7-12"

split_each_csv_file(path)

print('done')
