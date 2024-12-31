"""
Basic file operations in Python: reading and writing text files.
"""

def write_basic_file():
    """Demonstrate basic file writing."""
    # Writing to a file
    with open('sample.txt', 'w') as file:
        file.write("Hello, this is a sample file.\n")
        file.write("This is the second line.\n")
        file.write("And this is the third line.")
    
    print("File 'sample.txt' has been written.")

def read_basic_file():
    """Demonstrate different ways to read a file."""
    print("\nReading file contents:")
    
    # Read entire file as a string
    print("\nReading entire file:")
    with open('sample.txt', 'r') as file:
        content = file.read()
        print(content)
    
    # Read file line by line
    print("\nReading line by line:")
    with open('sample.txt', 'r') as file:
        for line in file:
            print(f"Line: {line.strip()}")
    
    # Read all lines into a list
    print("\nReading all lines into a list:")
    with open('sample.txt', 'r') as file:
        lines = file.readlines()
        print(f"Number of lines: {len(lines)}")
        for i, line in enumerate(lines, 1):
            print(f"Line {i}: {line.strip()}")

def append_to_file():
    """Demonstrate appending to a file."""
    # Append mode
    with open('sample.txt', 'a') as file:
        file.write("\nThis line was appended.")
        file.write("\nAnd so was this one.")
    
    print("\nContent after appending:")
    with open('sample.txt', 'r') as file:
        print(file.read())

# Example usage
if __name__ == "__main__":
    write_basic_file()
    read_basic_file()
    append_to_file() 