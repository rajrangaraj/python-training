"""
Collection data types in Python: lists, tuples, and sets.
"""

# Lists
numbers = [1, 2, 3, 4, 5]
print("List Operations:")
print(f"List: {numbers}, Type: {type(numbers)}")
print(f"First element: {numbers[0]}")
print(f"Last element: {numbers[-1]}")
print(f"Slicing: {numbers[1:3]}")
numbers.append(6)
print(f"After append: {numbers}")

# Tuples (immutable lists)
coordinates = (10, 20)
print("\nTuple Operations:")
print(f"Tuple: {coordinates}, Type: {type(coordinates)}")
x, y = coordinates  # tuple unpacking
print(f"Unpacked coordinates: x={x}, y={y}")

# Sets
unique_numbers = {1, 2, 3, 3, 4, 4, 5}  # duplicates are removed
print("\nSet Operations:")
print(f"Set: {unique_numbers}, Type: {type(unique_numbers)}")
other_numbers = {4, 5, 6, 7}
print(f"Union: {unique_numbers | other_numbers}")
print(f"Intersection: {unique_numbers & other_numbers}") 