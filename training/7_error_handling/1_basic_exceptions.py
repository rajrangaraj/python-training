"""
Basic exception handling in Python using try-except blocks.
"""

# Basic try-except
def basic_division(a, b):
    """Demonstrate basic error handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None

# Multiple except blocks
def convert_to_number(value):
    """Handle multiple types of exceptions."""
    try:
        number = float(value)
        return number
    except ValueError:
        print("Error: Invalid number format!")
    except TypeError:
        print("Error: Invalid type!")
    return None

# Try-except-else
def read_number_from_user():
    """Demonstrate try-except-else usage."""
    try:
        number = int(input("Enter a number: "))
    except ValueError:
        print("That's not a valid number!")
        return None
    else:
        print("Successfully read the number!")
        return number

# Try-except-finally
def open_and_read_file(filename):
    """Demonstrate try-except-finally with file handling."""
    file = None
    try:
        file = open(filename, 'r')
        content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return None
    finally:
        if file:
            file.close()
            print("File closed successfully")

# Example usage
if __name__ == "__main__":
    print("Basic Division Example:")
    print(f"10 / 2 = {basic_division(10, 2)}")
    print(f"10 / 0 = {basic_division(10, 0)}")
    
    print("\nType Conversion Example:")
    print(f"Converting '123': {convert_to_number('123')}")
    print(f"Converting 'abc': {convert_to_number('abc')}")
    print(f"Converting None: {convert_to_number(None)}")
    
    print("\nUser Input Example:")
    number = read_number_from_user()
    
    print("\nFile Handling Example:")
    content = open_and_read_file("nonexistent.txt") 