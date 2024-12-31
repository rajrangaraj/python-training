"""
Basic arithmetic operations in Python.
"""

# Basic operations with integers
a = 10
b = 3

print("Basic Integer Operations:")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Integer Division: {a} // {b} = {a // b}")
print(f"Modulus: {a} % {b} = {a % b}")

# Operations with floats
x = 10.5
y = 3.2

print("\nFloat Operations:")
print(f"Addition: {x} + {y} = {x + y}")
print(f"Subtraction: {x} - {y} = {x - y}")
print(f"Multiplication: {x} * {y} = {x * y}")
print(f"Division: {x} / {y} = {x / y}")

# Combined operations
print("\nCombined Operations:")
result = a + b * 2
print(f"{a} + {b} * 2 = {result}")
result = (a + b) * 2
print(f"({a} + {b}) * 2 = {result}")

# Operations with negative numbers
print("\nOperations with Negative Numbers:")
print(f"Negation of {a}: {-a}")
print(f"Absolute value of -10: {abs(-10)}") 