"""
Demonstration of different file modes and their uses in Python.
"""

def demonstrate_write_modes():
    """Show different write modes: 'w', 'a', 'x'."""
    # Write mode ('w') - Creates new file or overwrites existing
    print("Write mode example:")
    with open('write_test.txt', 'w') as file:
        file.write("This is line 1\n")
        file.write("This is line 2\n")
    print("File written in write mode")
    
    # Append mode ('a') - Adds to existing file
    print("\nAppend mode example:")
    with open('write_test.txt', 'a') as file:
        file.write("This line is appended\n")
        file.write("So is this one\n")
    print("Content appended to file")
    
    # Exclusive creation ('x') - Only if file doesn't exist
    print("\nExclusive creation mode example:")
    try:
        with open('new_file.txt', 'x') as file:
            file.write("This is a new file\n")
        print("New file created")
    except FileExistsError:
        print("File already exists!")

def demonstrate_read_modes():
    """Show different read modes: 'r', 'rb'."""
    # Text mode ('r')
    print("\nText mode reading:")
    with open('write_test.txt', 'r') as file:
        content = file.read()
        print(content)
    
    # Binary mode ('rb')
    print("\nBinary mode reading:")
    with open('write_test.txt', 'rb') as file:
        binary_content = file.read()
        print(f"Binary content (first 10 bytes): {binary_content[:10]}")

def demonstrate_plus_modes():
    """Show read and write modes: 'r+', 'w+'."""
    # Read and write mode ('r+')
    print("\nRead and write mode example:")
    with open('write_test.txt', 'r+') as file:
        content = file.read()
        print("Current content:", content)
        file.seek(0)  # Go to start of file
        file.write("New first line\n")  # Overwrite from current position
    
    # Write and read mode ('w+')
    print("\nWrite and read mode example:")
    with open('write_test.txt', 'w+') as file:
        file.write("Complete new content\n")
        file.seek(0)  # Go to start to read
        new_content = file.read()
        print("New content:", new_content)

def demonstrate_seek_tell():
    """Show file position operations with seek() and tell()."""
    print("\nFile position example:")
    with open('write_test.txt', 'r+') as file:
        # Get current position
        print(f"Initial position: {file.tell()}")
        
        # Read first line
        first_line = file.readline()
        print(f"First line: {first_line.strip()}")
        print(f"After reading first line: {file.tell()}")
        
        # Seek to specific position
        file.seek(0)  # Go to start
        print(f"After seeking to start: {file.tell()}")
        
        # Seek from current position
        file.seek(5, 1)  # Go 5 bytes forward from current position
        print(f"After seeking 5 bytes forward: {file.tell()}")
        
        # Seek from end
        file.seek(0, 2)  # Go to end
        print(f"After seeking to end: {file.tell()}")

# Example usage
if __name__ == "__main__":
    demonstrate_write_modes()
    demonstrate_read_modes()
    demonstrate_plus_modes()
    demonstrate_seek_tell() 