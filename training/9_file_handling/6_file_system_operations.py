"""
Demonstration of file system operations using os and shutil modules.
"""

import os
import shutil
from datetime import datetime

def demonstrate_path_operations():
    """Show common path operations."""
    print("Path Operations:")
    
    # Current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Join paths
    new_dir = os.path.join(current_dir, 'test_folder')
    print(f"New directory path: {new_dir}")
    
    # Path components
    path = '/home/user/documents/file.txt'
    print(f"\nPath components for: {path}")
    print(f"Directory: {os.path.dirname(path)}")
    print(f"Filename: {os.path.basename(path)}")
    print(f"Extension: {os.path.splitext(path)[1]}")

def demonstrate_directory_operations():
    """Show directory creation and listing operations."""
    print("\nDirectory Operations:")
    
    # Create directory
    os.makedirs('test_folder/subfolder', exist_ok=True)
    print("Created test_folder and subfolder")
    
    # List directory contents
    print("\nDirectory contents:")
    for item in os.listdir('test_folder'):
        item_path = os.path.join('test_folder', item)
        item_type = 'Directory' if os.path.isdir(item_path) else 'File'
        print(f"{item} - {item_type}")
    
    # Walk directory tree
    print("\nDirectory tree:")
    for root, dirs, files in os.walk('test_folder'):
        level = root.replace('test_folder', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

def demonstrate_file_operations():
    """Show file operations and information."""
    print("\nFile Operations:")
    
    # Create test file
    test_file = 'test_folder/test.txt'
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    # File information
    stats = os.stat(test_file)
    print(f"\nFile information for {test_file}:")
    print(f"Size: {stats.st_size} bytes")
    print(f"Created: {datetime.fromtimestamp(stats.st_ctime)}")
    print(f"Modified: {datetime.fromtimestamp(stats.st_mtime)}")
    
    # Copy file
    shutil.copy2(test_file, 'test_folder/test_backup.txt')
    print("\nFile copied")
    
    # Rename file
    os.rename('test_folder/test_backup.txt', 'test_folder/test_new.txt')
    print("File renamed")

def demonstrate_file_patterns():
    """Show file pattern matching and filtering."""
    import glob
    
    print("\nFile Pattern Matching:")
    
    # Create some test files
    test_files = ['test1.txt', 'test2.txt', 'data.csv', 'image.jpg']
    for file in test_files:
        with open(os.path.join('test_folder', file), 'w') as f:
            f.write("Test content")
    
    # Find files by pattern
    print("\nAll .txt files:")
    for file in glob.glob('test_folder/*.txt'):
        print(file)
    
    print("\nFiles starting with 'test':")
    for file in glob.glob('test_folder/test*'):
        print(file)

def cleanup_test_files():
    """Clean up test files and directories."""
    try:
        shutil.rmtree('test_folder')
        print("\nTest files and directories cleaned up")
    except Exception as e:
        print(f"\nError during cleanup: {e}")

# Example usage
if __name__ == "__main__":
    demonstrate_path_operations()
    demonstrate_directory_operations()
    demonstrate_file_operations()
    demonstrate_file_patterns()
    cleanup_test_files() 