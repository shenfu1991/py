import pandas as pd
import os

# The directory where the CSV files are stored
directory = '/Users/xuanyuan/Documents/csv'

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Full path to the CSV file
        filepath = os.path.join(directory, filename)
        
        # Load the data from the CSV file
        data = pd.read_csv(filepath)

        # Remove 'result' column from consideration
        data_without_result = data.drop(columns=['result'])

        # Create a copy of the original DataFrame to preserve the original data
        data_cleaned = data.copy()

        # Shift the DataFrame down by one row
        data_shifted = data_without_result.shift(-1)

        # Count the number of columns where the original and shifted DataFrames have the same value
        num_same_values = (data_without_result == data_shifted).sum(axis=1)

        # Find the rows where 4 or more columns have the same value in the original and shifted DataFrames
        rows_to_remove = num_same_values >= 4

        # Remove these rows from the cleaned DataFrame
        data_cleaned = data_cleaned[~rows_to_remove]

        # Save the cleaned data to a new CSV file
        cleaned_filepath = os.path.join(directory, 'cleaned_' + filename)
        data_cleaned.to_csv(cleaned_filepath, index=False)

        # Print the shape of the cleaned DataFrame
        print(f'{filename}: {data_cleaned.shape}')
