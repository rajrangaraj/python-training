"""
Advanced mathematical operations in Python.
"""

# Exponentiation
base = 2
exponent = 3
print("Exponentiation:")
print(f"{base}^{exponent} = {base ** exponent}")
print(f"Using pow(): {pow(base, exponent)}")

# Rounding operations
pi = 3.14159
print("\nRounding Operations:")
print(f"Original number: {pi}")
print(f"Round to 2 decimal places: {round(pi, 2)}")
print(f"Round to nearest integer: {round(pi)}")
print(f"Floor division: {pi // 1}")
print(f"Integer conversion: {int(pi)}")

# Absolute values
numbers = [5, -10, 3.14, -2.5]
print("\nAbsolute Values:")
for num in numbers:
    print(f"|{num}| = {abs(num)}")

# Complex numbers
c1 = 3 + 4j
c2 = 2 + 3j
print("\nComplex Numbers:")
print(f"Complex number: {c1}")
print(f"Real part: {c1.real}")
print(f"Imaginary part: {c1.imag}")
print(f"Addition: {c1} + {c2} = {c1 + c2}")
print(f"Multiplication: {c1} * {c2} = {c1 * c2}") 