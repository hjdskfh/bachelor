import pandas as pd
import csv
import ast
import os
import datetime


def find_max_skr(file_path):
    max_skr = float('-inf')  # Initialize with negative infinity
    max_skr_row = None  # To store the row with the maximum SKR

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row
        print("Header:", header)  # Print the header

        for row in reader:
            try:
                # Check if the last column is a stringified list or a float
                if isinstance(row[-1], str) and row[-1].startswith("[") and row[-1].endswith("]"):
                    skr_value = ast.literal_eval(row[-1])[0]  # Parse the stringified list
                else:
                    skr_value = float(row[-1])  # Directly convert to float if it's not a list

                if skr_value > max_skr:
                    max_skr = skr_value
                    max_skr_row = row
            except (ValueError, IndexError, SyntaxError, TypeError):
                # Handle any parsing errors or malformed rows
                print(f"Skipping malformed row: {row}")

    return max_skr, max_skr_row

# Directory containing the CSV files
csv_dir =  r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_05_06'
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
summary_file = os.path.join(csv_dir, f"max_skr_summary_{timestamp}.csv")

# Process all CSV files in the directory
for file_name in os.listdir(csv_dir):
    if file_name.endswith('.csv'):  # Only process CSV files
        file_path = os.path.join(csv_dir, file_name)
        max_skr, max_skr_row = find_max_skr(file_path)
        print(f"File: {file_name}")
        print(f"Max SKR: {max_skr}")
        print(f"Row with Max SKR: {max_skr_row}\n")

results = []
for file_name in os.listdir(csv_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_dir, file_name)
        max_skr, max_skr_row = find_max_skr(file_path)
        results.append({
            'File': file_name,
            'Max SKR': max_skr,
            'Row': max_skr_row
        })

# Save results to a CSV file
summary_file_excel = os.path.join(csv_dir, f"max_skr_summary_{timestamp}.xlsx")
summary_df = pd.DataFrame(results)
summary_df.to_excel(summary_file_excel, index=False)
print(f"Summary saved to {summary_file_excel}")