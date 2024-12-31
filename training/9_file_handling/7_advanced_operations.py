"""
Advanced file handling operations and techniques.
"""

import os
import io
import tempfile
import filecmp
from contextlib import contextmanager

def demonstrate_string_io():
    """Show usage of StringIO for in-memory file operations."""
    print("StringIO Example:")
    
    # Write to string buffer
    output = io.StringIO()
    output.write('First line\n')
    output.write('Second line\n')
    
    # Get value and show position
    print(f"Current position: {output.tell()}")
    content = output.getvalue()
    print(f"Content:\n{content}")
    
    # Read from string buffer
    output.seek(0)  # Go to start
    print("\nReading line by line:")
    for line in output:
        print(f"Line: {line.strip()}")
    
    output.close()

def demonstrate_binary_io():
    """Show usage of BytesIO for binary data."""
    print("\nBytesIO Example:")
    
    # Write binary data
    binary_data = io.BytesIO()
    binary_data.write(b'Hello\n')
    binary_data.write(b'Binary\n')
    binary_data.write(b'World')
    
    # Read binary data
    binary_data.seek(0)
    content = binary_data.read()
    print(f"Binary content: {content}")
    print(f"Decoded content: {content.decode()}")
    
    binary_data.close()

@contextmanager
def temporary_file_context():
    """Context manager for temporary file handling."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        yield temp_file
    finally:
        temp_file.close()
        os.unlink(temp_file.name)

def demonstrate_temp_files():
    """Show usage of temporary files."""
    print("\nTemporary File Example:")
    
    # Using context manager
    with temporary_file_context() as temp:
        # Write to temp file
        temp.write(b"Temporary content\n")
        temp.flush()
        
        print(f"Temp file created: {temp.name}")
        
        # Read from temp file
        with open(temp.name, 'rb') as f:
            content = f.read().decode()
            print(f"Content: {content}")
    
    print("Temp file cleaned up")

def demonstrate_file_comparison():
    """Show file comparison operations."""
    print("\nFile Comparison Example:")
    
    # Create test files
    with open('file1.txt', 'w') as f1:
        f1.write("Line 1\nLine 2\nLine 3\n")
    
    with open('file2.txt', 'w') as f2:
        f2.write("Line 1\nLine 2\nLine 3\n")
    
    with open('file3.txt', 'w') as f3:
        f3.write("Line 1\nLine 2\nDifferent Line 3\n")
    
    # Compare files
    print(f"file1 and file2 are identical: {filecmp.cmp('file1.txt', 'file2.txt')}")
    print(f"file1 and file3 are identical: {filecmp.cmp('file1.txt', 'file3.txt')}")
    
    # Clean up
    os.remove('file1.txt')
    os.remove('file2.txt')
    os.remove('file3.txt')

def demonstrate_file_buffering():
    """Show different file buffering modes."""
    print("\nFile Buffering Example:")
    
    # Unbuffered
    with open('unbuffered.txt', 'wb', buffering=0) as f:
        f.write(b"Unbuffered write")
    
    # Line buffered
    with open('line_buffered.txt', 'w', buffering=1) as f:
        f.write("Line 1\n")  # Flushes automatically
        f.write("Line 2\n")  # Flushes automatically
    
    # Custom buffer size (in bytes)
    with open('custom_buffered.txt', 'w', buffering=2048) as f:
        f.write("Custom buffered content")
    
    # Clean up
    for file in ['unbuffered.txt', 'line_buffered.txt', 'custom_buffered.txt']:
        os.remove(file)

# Example usage
if __name__ == "__main__":
    demonstrate_string_io()
    demonstrate_binary_io()
    demonstrate_temp_files()
    demonstrate_file_comparison()
    demonstrate_file_buffering() 