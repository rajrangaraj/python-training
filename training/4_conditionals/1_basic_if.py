"""
Basic if-else statements in Python.
"""

# Simple if statement
age = 18
if age >= 18:
    print("You are an adult")

# If-else statement
temperature = 25
if temperature > 30:
    print("It's hot outside")
else:
    print("It's not too hot")

# Multiple conditions with if-else
score = 85
if score >= 90:
    print("Grade: A")
else:
    print("Grade: B or lower")

# Comparing different types of values
name = "Alice"
age = 25
has_id = True

if name and age >= 21 and has_id:
    print("Can enter the venue")
else:
    print("Cannot enter the venue")

# Using 'in' operator
fruits = ["apple", "banana", "orange"]
if "apple" in fruits:
    print("We have apples!")

# Testing for None
value = None
if value is None:
    print("Value is None")
else:
    print("Value is not None")

# Boolean expressions
is_sunny = True
is_warm = True
if is_sunny and is_warm:
    print("Perfect day for a picnic!") 