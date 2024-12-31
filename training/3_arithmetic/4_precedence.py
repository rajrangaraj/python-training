"""
Operator precedence examples in Python.
"""

# Basic precedence examples
print("Basic Precedence:")
result = 2 + 3 * 4
print(f"2 + 3 * 4 = {result}")
result = (2 + 3) * 4
print(f"(2 + 3) * 4 = {result}")

# Complex precedence examples
print("\nComplex Precedence:")
a = 2
b = 3
c = 4
d = 5

result = a + b * c / d
print(f"a + b * c / d = {result}")
result = (a + b) * (c / d)
print(f"(a + b) * (c / d) = {result}")

# Precedence with exponents
print("\nExponent Precedence:")
result = 2 + 3 ** 2
print(f"2 + 3 ** 2 = {result}")
result = (2 + 3) ** 2
print(f"(2 + 3) ** 2 = {result}")

# Mixed operations
print("\nMixed Operations:")
result = 10 - 2 * 3 + 4 / 2
print(f"10 - 2 * 3 + 4 / 2 = {result}")
result = (10 - 2) * (3 + 4) / 2
print(f"(10 - 2) * (3 + 4) / 2 = {result}")

# Common precedence rules
print("\nPrecedence Rules Examples:")
print("1. Parentheses")
print("2. Exponents")
print("3. Multiplication/Division (left to right)")
print("4. Addition/Subtraction (left to right)") 