import difflib

def read_profiler_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()
    

def extract_significant_time_lines(lines):
    significant_lines = []
    for line in lines:
        if 'function calls' in line or 'filename:lineno(function)' in line:
            continue
        parts = line.split()
        if len(parts) > 5:
            try:
                cumtime = float(parts[4])
                if cumtime > 0.1:  # Set your threshold for significant time differences
                    significant_lines.append(line)
            except ValueError:
                continue
    return significant_lines

def alternate_lines(lines1, lines2):
    max_length = max(len(lines1), len(lines2))
    alternated_lines = []
    for i in range(max_length):
        if i < len(lines1):
            alternated_lines.append(f"File1: {lines1[i]}")
        if i < len(lines2):
            alternated_lines.append(f"File2: {lines2[i]}")
    return alternated_lines

def compare_profiler_files(file1, file2, output_file):
    lines1 = read_profiler_file(file1)
    lines2 = read_profiler_file(file2)
    
    sig_lines1 = extract_significant_time_lines(lines1)
    sig_lines2 = extract_significant_time_lines(lines2)
    
    alternated_lines = alternate_lines(sig_lines1, sig_lines2)
    
    with open(output_file, 'w') as file:
        for line in alternated_lines:
            file.write(line + '\n')

if __name__ == "__main__":
    file1 = 'code/additional/profile_output_10_3.txt'  # Path to the first profiler output file
    file2 = 'code/additional/profile_output_11_0.txt'  # Path to the second profiler output file
    output_file = 'code/additional/diff_output.txt'  # Path to the output file for differences
    
    compare_profiler_files(file1, file2, output_file)
    print(f"Differences have been saved to {output_file}")