"""
Demonstration of lambda functions (anonymous functions) in Python.
"""

# Basic lambda function
square = lambda x: x ** 2

# Lambda with multiple parameters
add = lambda x, y: x + y

# Lambda with conditional expression
is_even = lambda x: True if x % 2 == 0 else False

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

# Lambda in sorting
points = [(1, 2), (3, 1), (2, 4)]
sorted_by_y = sorted(points, key=lambda point: point[1])

# Lambda with reduce
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)

# Lambda in dictionary sorting
people = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 20}
]
sorted_by_age = sorted(people, key=lambda person: person["age"])

# Example usage
if __name__ == "__main__":
    print("Basic lambda usage:")
    print(f"Square of 5: {square(5)}")
    print(f"Add 3 and 4: {add(3, 4)}")
    
    print("\nConditional lambda:")
    print(f"Is 4 even? {is_even(4)}")
    print(f"Is 5 even? {is_even(5)}")
    
    print("\nLambda with map and filter:")
    print(f"Original numbers: {numbers}")
    print(f"Squared numbers: {squared_numbers}")
    print(f"Even numbers: {even_numbers}")
    
    print("\nSorting with lambda:")
    print(f"Original points: {points}")
    print(f"Sorted by y-coordinate: {sorted_by_y}")
    
    print("\nReduce with lambda:")
    print(f"Product of numbers: {product}")
    
    print("\nDictionary sorting:")
    print("Sorted by age:")
    for person in sorted_by_age:
        print(f"{person['name']}: {person['age']}") 