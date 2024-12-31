"""
Introduction to Python modules and basic module usage.
"""

# Import entire module
import math

# Import specific items from a module
from random import randint, choice

# Import with alias
import datetime as dt

# Import all items (not recommended)
from string import *

def demonstrate_math_module():
    """Show basic usage of the math module."""
    print("Math Module Examples:")
    print(f"Pi: {math.pi}")
    print(f"Square root of 16: {math.sqrt(16)}")
    print(f"Cosine of 0: {math.cos(0)}")
    print(f"Factorial of 5: {math.factorial(5)}")

def demonstrate_random_module():
    """Show basic usage of random module functions."""
    print("\nRandom Module Examples:")
    print(f"Random number between 1 and 10: {randint(1, 10)}")
    
    fruits = ["apple", "banana", "orange", "grape"]
    print(f"Random fruit: {choice(fruits)}")

def demonstrate_datetime_module():
    """Show basic usage of datetime module."""
    print("\nDatetime Module Examples:")
    current_time = dt.datetime.now()
    print(f"Current time: {current_time}")
    print(f"Current year: {current_time.year}")
    print(f"Formatted date: {current_time.strftime('%Y-%m-%d')}")

def demonstrate_string_module():
    """Show basic usage of string module constants."""
    print("\nString Module Examples:")
    print(f"ASCII letters: {ascii_letters}")
    print(f"Digits: {digits}")
    print(f"Punctuation: {punctuation}")

# Example usage
if __name__ == "__main__":
    demonstrate_math_module()
    demonstrate_random_module()
    demonstrate_datetime_module()
    demonstrate_string_module() 