import pandas as pd

# Load the two files into DataFrames
file1 = pd.read_csv(r"C:\Users\leavi\bachelor\deadtime_SKR_results_20250427_1528_tot_sym_4000000.csv")
file2 = pd.read_csv(r"C:\Users\leavi\bachelor\OG_SKR_results_20250427_1527_tot_sym_1460000.0.csv")

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