import pandas as pd
import os

def correct_sequence(df):
    corrected_df = df.copy()
    changes = 0
    changed_indices = []

    for i in range(len(df) - 3):
        sequence = corrected_df['result'].iloc[i:i+4].tolist()

        # Check for patterns and correct them
        if sequence in [['short', 'short', 'long', 'short'], ['short', 'long', 'short', 'short']]:
            if sequence[2] != 'short':
                changes += 1
                changed_indices.append(i+2)
                corrected_df.at[i+2, 'result'] = 'short'
        elif sequence in [['long', 'long', 'short', 'long'], ['long', 'short', 'long', 'long']]:
            if sequence[2] != 'long':
                changes += 1
                changed_indices.append(i+2)
                corrected_df.at[i+2, 'result'] = 'long'
        
    return corrected_df, changes, changed_indices

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Load the CSV file
                df = pd.read_csv(file_path)

                # Correct the sequence
                corrected_df, changes, changed_indices = correct_sequence(df)

                # If there were changes, overwrite the file and print the changes
                if changes > 0:
                    corrected_df.to_csv(file_path, index=False)
                    print(f"Modified file: {file_path}")
                    print(f"Total changes: {changes}")
                    # print(f"Changed rows: {changed_indices}")
                else:
                    print(f"No changes in file: {file_path}")

# Specify the folder path (you might need to adjust this based on your local setup)
# folder_path = '/path/to/your/folder'
# process_folder(folder_path)


# Specify the folder path (you might need to adjust this based on your local setup)
folder_path = '/Users/xuanyuan/Downloads/3d'
process_folder(folder_path)
