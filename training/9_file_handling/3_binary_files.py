"""
Working with binary files in Python.
"""

import struct
import array

def write_binary_numbers():
    """Write numbers in binary format."""
    # Write integers in binary format
    numbers = [1, 2, 3, 4, 5]
    with open('numbers.bin', 'wb') as file:
        # Pack numbers as 32-bit integers
        for num in numbers:
            file.write(struct.pack('i', num))
    print("Binary numbers written to file")

def read_binary_numbers():
    """Read binary format numbers."""
    numbers = []
    with open('numbers.bin', 'rb') as file:
        # Read and unpack 32-bit integers
        while True:
            data = file.read(4)  # Read 4 bytes (32 bits)
            if not data:
                break
            number = struct.unpack('i', data)[0]
            numbers.append(number)
    print(f"Read numbers: {numbers}")

def write_binary_array():
    """Write array of numbers in binary format."""
    # Create array of integers
    numbers = array.array('i', [10, 20, 30, 40, 50])
    with open('array.bin', 'wb') as file:
        numbers.tofile(file)
    print("Binary array written to file")

def read_binary_array():
    """Read binary format array."""
    numbers = array.array('i')
    with open('array.bin', 'rb') as file:
        numbers.fromfile(file, 5)  # Read 5 integers
    print(f"Read array: {numbers.tolist()}")

def write_binary_struct():
    """Write structured binary data."""
    # Define format: int, float, string (10 chars)
    format_string = 'if10s'
    data = [
        (1, 3.14, b'Hello     '),
        (2, 2.718, b'World     ')
    ]
    
    with open('struct.bin', 'wb') as file:
        for record in data:
            packed_data = struct.pack(format_string, *record)
            file.write(packed_data)
    print("Binary structures written to file")

def read_binary_struct():
    """Read structured binary data."""
    format_string = 'if10s'
    record_size = struct.calcsize(format_string)
    records = []
    
    with open('struct.bin', 'rb') as file:
        while True:
            data = file.read(record_size)
            if not data:
                break
            record = struct.unpack(format_string, data)
            # Convert bytes to string and strip padding
            record = (record[0], record[1], record[2].strip().decode())
            records.append(record)
    
    print("Read structures:")
    for record in records:
        print(f"ID: {record[0]}, Value: {record[1]}, Text: {record[2]}")

# Example usage
if __name__ == "__main__":
    print("Binary Numbers Example:")
    write_binary_numbers()
    read_binary_numbers()
    
    print("\nBinary Array Example:")
    write_binary_array()
    read_binary_array()
    
    print("\nBinary Structure Example:")
    write_binary_struct()
    read_binary_struct() 