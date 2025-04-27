import pandas as pd
import csv
import ast

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

# Load the two files into DataFrames
filepath_1 = r"C:\Users\leavi\bachelor\deadtime_SKR_results_20250427_1528_tot_sym_4000000.csv"
filepath_2 = r"C:\Users\leavi\bachelor\OG_SKR_results_20250427_1527_tot_sym_1460000.0.csv"
file1 = pd.read_csv(filepath_1)
file2 = pd.read_csv(filepath_2)

max_skr_file1, max_skr_row_file1 = find_max_skr(filepath_1)
max_skr_file2, max_skr_row_file2 = find_max_skr(filepath_2)
print(f"Max SKR in file 1: {max_skr_file1}, Row: {max_skr_row_file1}")
print(f"Max SKR in file 2: {max_skr_file2}, Row: {max_skr_row_file2}")

# Select the relevant columns for comparison
columns_to_compare = ["length_multiply", "mpn_s", "mpn_d", "desired_p_decoy", "desired_p_z_alice", "factor_x_mud"]

# Merge the two DataFrames on the selected columns
merged = pd.merge(file1, file2, on=columns_to_compare, suffixes=("_file1", "_file2"))

# Filter rows where the SKR values differ
skr_differences = merged[merged["SKR_file1"] != merged["SKR_file2"]]

# Display the rows with differences
print("Rows with differing SKR values:")
print(skr_differences)

# Optionally, save the differences to a new CSV file
skr_differences.to_csv("skr_differences.csv", index=False)

