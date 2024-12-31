"""
Using Python's math module for advanced mathematical operations.
"""

import math

# Constants
print("Mathematical Constants:")
print(f"π (pi): {math.pi}")
print(f"e: {math.e}")
print(f"tau (2π): {math.tau}")

# Trigonometric functions
angle = math.pi / 4  # 45 degrees
print("\nTrigonometric Functions:")
print(f"sin(π/4): {math.sin(angle)}")
print(f"cos(π/4): {math.cos(angle)}")
print(f"tan(π/4): {math.tan(angle)}")

# Logarithmic functions
n = 100
print("\nLogarithmic Functions:")
print(f"Natural log of {n}: {math.log(n)}")
print(f"Log base-10 of {n}: {math.log10(n)}")
print(f"Log base-2 of {n}: {math.log2(n)}")

# Square root and power functions
print("\nRoot and Power Functions:")
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Cube root of 27: {math.pow(27, 1/3)}")
print(f"2^10: {math.pow(2, 10)}")

# Ceiling and floor
x = 3.7
print("\nCeiling and Floor:")
print(f"Ceiling of {x}: {math.ceil(x)}")
print(f"Floor of {x}: {math.floor(x)}")

# Angular conversion
degrees = 45
radians = math.radians(degrees)
print("\nAngle Conversion:")
print(f"{degrees} degrees = {radians} radians")
print(f"{radians} radians = {math.degrees(radians)} degrees") 