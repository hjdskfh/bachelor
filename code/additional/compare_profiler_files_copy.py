def read_profiler_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

def extract_ncalls(lines):
    ncalls_lines = []
    for line in lines:
        if 'function calls' in line or 'filename:lineno(function)' in line:
            continue
        parts = line.split()
        if len(parts) > 0:
            try:
                ncalls = int(parts[0].replace(',', ''))  # Remove commas and convert to integer
                ncalls_lines.append((ncalls, line))  # Store ncalls along with the line for reference
            except ValueError:
                continue
    return ncalls_lines

def find_differences(lines1, lines2):
    differences = []
    ncalls_dict1 = {line[1]: line[0] for line in lines1}  # Map line to ncalls for file1
    ncalls_dict2 = {line[1]: line[0] for line in lines2}  # Map line to ncalls for file2

    all_lines = set(ncalls_dict1.keys()).union(set(ncalls_dict2.keys()))

    for line in all_lines:
        ncalls1 = ncalls_dict1.get(line, None)
        ncalls2 = ncalls_dict2.get(line, None)
        if ncalls1 != ncalls2:  # Only include lines with differences in ncalls
            differences.append((ncalls1, ncalls2, line))

    return differences

def compare_profiler_files(file1, file2, output_file):
    lines1 = read_profiler_file(file1)
    lines2 = read_profiler_file(file2)

    ncalls_lines1 = extract_ncalls(lines1)
    ncalls_lines2 = extract_ncalls(lines2)

    differences = find_differences(ncalls_lines1, ncalls_lines2)

    with open(output_file, 'w') as file:
        for ncalls1, ncalls2, line in differences:
            file.write(f"File1 ncalls: {ncalls1 if ncalls1 is not None else 'N/A'}\n")
            file.write(f"File2 ncalls: {ncalls2 if ncalls2 is not None else 'N/A'}\n")
            file.write(f"Line: {line}\n\n")

if __name__ == "__main__":
    file1 = 'code/additional/profile_output_10_3.txt'  # Path to the first profiler output file
    file2 = 'code/additional/profile_output_11_0.txt'  # Path to the second profiler output file
    output_file = 'code/additional/diff_output.txt'  # Path to the output file for differences

    compare_profiler_files(file1, file2, output_file)
    print(f"Differences have been saved to {output_file}")
