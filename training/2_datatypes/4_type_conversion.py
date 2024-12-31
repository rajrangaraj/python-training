"""
Type conversion (casting) between different Python data types.
"""

# String to number conversions
str_num = "42"
int_num = int(str_num)
float_num = float(str_num)
print(f"String to int: {str_num} -> {int_num}, Type: {type(int_num)}")
print(f"String to float: {str_num} -> {float_num}, Type: {type(float_num)}")

# Number to string conversion
num = 3.14159
str_pi = str(num)
print(f"\nNumber to string: {num} -> {str_pi}, Type: {type(str_pi)}")

# List, tuple, and set conversions
my_list = [1, 2, 2, 3, 3, 4]
my_tuple = tuple(my_list)
my_set = set(my_list)
print(f"\nList to tuple: {my_tuple}, Type: {type(my_tuple)}")
print(f"List to set (removes duplicates): {my_set}, Type: {type(my_set)}")

# Boolean conversions
print("\nBoolean conversions:")
print(f"Empty string to bool: bool('') = {bool('')}")
print(f"Non-empty string to bool: bool('hello') = {bool('hello')}")
print(f"Zero to bool: bool(0) = {bool(0)}")
print(f"Non-zero to bool: bool(42) = {bool(42)}")

# Dictionary conversions
pairs = [("name", "John"), ("age", 30)]
dict_from_pairs = dict(pairs)
print(f"\nList of tuples to dictionary: {dict_from_pairs}") 