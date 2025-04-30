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
# filepath_1 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\results_1_20250429_20.csv"  #0.1
# filepath_2 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\results_2_20250429_20.csv"
# filepath_3 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\results_7_20250429_12.csv"
# filepath_4 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\results_4_20250429_20.csv"

# file1 = pd.read_csv(filepath_1)
# file2 = pd.read_csv(filepath_2)
# file3 = pd.read_csv(filepath_3)
# file4 = pd.read_csv(filepath_4)

# max_skr_file1, max_skr_row_file1 = find_max_skr(filepath_1)
# max_skr_file2, max_skr_row_file2 = find_max_skr(filepath_2)
# max_skr_file3, max_skr_row_file3 = find_max_skr(filepath_3)
# max_skr_file4, max_skr_row_file4 = find_max_skr(filepath_4)
# print(f"Max SKR in file 1: {max_skr_file1}, Row: {max_skr_row_file1}")
# print(f"Max SKR in file 2: {max_skr_file2}, Row: {max_skr_row_file2}")
# print(f"Max SKR in file 3: {max_skr_file3}, Row: {max_skr_row_file3}")
# print(f"Max SKR in file 4: {max_skr_file4}, Row: {max_skr_row_file4}")


# extinction ratio abend 18 uhr run nr 8
# filepath_8 = r'C:\Users\leavi\bachelor\wichtig\extinction_ratio\results_8_20250429_20.csv'
# file8 = pd.read_csv(filepath_8)
# max_skr_file8, max_skr_row_file8 = find_max_skr(filepath_8)
# print(f"Max SKR in file 8: {max_skr_file8}, Row: {max_skr_row_file8}")

# estinction ratio 10 uhr 9 10 11
# filepath_9 = r'C:\Users\leavi\bachelor\wichtig\extinction_ratio\results_9_20250429_23.csv'
# file9 = pd.read_csv(filepath_9)
# max_skr_file9, max_skr_row_file9 = find_max_skr(filepath_9)
# print(f"Max SKR in file 9: {max_skr_file9}, Row: {max_skr_row_file9}")
# filepath_10 = r'C:\Users\leavi\bachelor\wichtig\extinction_ratio\results_10_20250429_23.csv'
# file10 = pd.read_csv(filepath_10)
# max_skr_file10, max_skr_row_file10 = find_max_skr(filepath_10)
# print(f"Max SKR in file 10: {max_skr_file10}, Row: {max_skr_row_file10}")
# filepath_11 = r'C:\Users\leavi\bachelor\wichtig\extinction_ratio\results_11_20250429_23.csv'
# file11 = pd.read_csv(filepath_11)
# max_skr_file11, max_skr_row_file11 = find_max_skr(filepath_11)
# print(f"Max SKR in file 11: {max_skr_file11}, Row: {max_skr_row_file11}")

# nachtmessung auf mittwoch
# filepath_1 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\new_results_1_20250430_10.csv"  #0.1
# filepath_2 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\new_results_2_20250430_10.csv"
# filepath_3 = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen\new_results_3_20250430_10.csv"

# file1 = pd.read_csv(filepath_1)
# file2 = pd.read_csv(filepath_2)
# file3 = pd.read_csv(filepath_3)

# max_skr_file1, max_skr_row_file1 = find_max_skr(filepath_1)
# max_skr_file2, max_skr_row_file2 = find_max_skr(filepath_2)
# max_skr_file3, max_skr_row_file3 = find_max_skr(filepath_3)
# print(f"Max SKR in file 1: {max_skr_file1}, Row: {max_skr_row_file1}")
# print(f"Max SKR in file 2: {max_skr_file2}, Row: {max_skr_row_file2}")
# print(f"Max SKR in file 3: {max_skr_file3}, Row: {max_skr_row_file3}")


# voltages 5 & 6
# filepath_5 = r'C:\Users\leavi\bachelor\wichtig\voltage_messungen\results_5_20250429_21.csv'
# file5 = pd.read_csv(filepath_5)
# max_skr_file5, max_skr_row_file5 = find_max_skr(filepath_5)
# print(f"Max SKR in file 5: {max_skr_file5}, Row: {max_skr_row_file5}")
# filepath_6 = r'C:\Users\leavi\bachelor\wichtig\voltage_messungen\results_6_20250429_21.csv'
# file6 = pd.read_csv(filepath_6)
# max_skr_file6, max_skr_row_file6 = find_max_skr(filepath_6)
# print(f"Max SKR in file 6: {max_skr_file6}, Row: {max_skr_row_file6}")



# # Select the relevant columns for comparison
# columns_to_compare = ["length_multiply", "mpn_s", "mpn_d", "desired_p_decoy", "desired_p_z_alice", "factor_x_mud"]

# # Merge the two DataFrames on the selected columns
# merged = pd.merge(file1, file2, on=columns_to_compare, suffixes=("_file1", "_file2"))

# # Filter rows where the SKR values differ
# skr_differences = merged[merged["SKR_file1"] != merged["SKR_file2"]]

# # Display the rows with differences
# print("Rows with differing SKR values:")
# print(skr_differences)

# # Optionally, save the differences to a new CSV file
# skr_differences.to_csv("skr_differences.csv", index=False)


# add up counts
import os
import re

# Define the folder containing the .txt files
input_dir = r"C:\Users\leavi\bachelor\wichtig\gute_Messungen"

# Updated regular expression to match the desired values
pattern = re.compile(
    r"n_Z_mus_in:\s*\[(.*?)\],\s*n_Z_mud_in:\s*\[(.*?)\],\s*n_X_mus_in:\s*\[(.*?)\],\s*n_X_mud_in:\s*\[(.*?)\],\s*"
    r"m_Z_mus_in:\s*\[(.*?)\],\s*m_Z_mud_in:\s*\[(.*?)\],\s*m_X_mus_in:\s*\[(.*?)\],\s*m_X_mud_in:\s*\[(.*?)\]"
)

# Iterate through all files in the folder
for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Processing file: {file_name}")
            match = pattern.search(content)
            if match:
                print(f"Match found in {file_name}")

                # Extract the values
                n_Z_mus_in = float(match.group(1))
                n_Z_mud_in = float(match.group(2))
                n_X_mus_in = float(match.group(3))
                n_X_mud_in = float(match.group(4))
                m_Z_mus_in = float(match.group(5))
                m_Z_mud_in = float(match.group(6))
                m_X_mus_in = float(match.group(7))
                m_X_mud_in = float(match.group(8))

                # Calculate the sum
                total_sum = n_Z_mus_in + n_Z_mud_in + n_X_mus_in + n_X_mud_in + m_Z_mus_in + m_Z_mud_in + m_X_mus_in + m_X_mud_in

                # Save the result to a new file with a similar name
                output_file_name = f"{os.path.splitext(file_name)[0]}_sum.txt"
                output_file_path = os.path.join(input_dir, output_file_name)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"File: {file_name}\n")
                    output_file.write(f"Sum: {total_sum}\n")

                # Print confirmation
                print(f"Processed {file_name}, saved result to {output_file_name}")
            else:
                print(f"No match found in {file_name}")