import psutil

def check_free_memory():
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024 ** 2)  # Convert bytes to MB
    used_memory = mem.used / (1024 ** 2)  # Used memory in MB
    available_memory = mem.available / (1024 ** 2)  # Memory that can still be used
    free_memory = mem.free / (1024 ** 2)  # Memory completely unused (not allocated at all)
    
    print(f"Total Memory: {total_memory:.2f} MB")
    print(f"Used Memory: {used_memory:.2f} MB")
    print(f"Available Memory: {available_memory:.2f} MB")
    print(f"Free Memory: {free_memory:.2f} MB")

    return available_memory

free_memory = check_free_memory()