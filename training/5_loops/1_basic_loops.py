"""
Basic loop structures in Python: for and while loops.
"""

# Simple for loop with range
print("Basic counting:")
for i in range(5):
    print(f"Count: {i}")

# For loop with start, stop, and step
print("\nCounting by twos:")
for i in range(0, 10, 2):
    print(f"Number: {i}")

# For loop with list
print("\nIterating through a list:")
fruits = ["apple", "banana", "orange", "grape"]
for fruit in fruits:
    print(f"Current fruit: {fruit}")

# For loop with enumeration
print("\nEnumerated iteration:")
for index, fruit in enumerate(fruits, 1):
    print(f"Fruit #{index}: {fruit}")

# Basic while loop
print("\nWhile loop example:")
counter = 0
while counter < 5:
    print(f"Counter is: {counter}")
    counter += 1

# While loop with user input
print("\nInteractive while loop:")
while True:
    response = input("Type 'quit' to exit: ")
    if response.lower() == 'quit':
        print("Exiting loop...")
        break
    print(f"You typed: {response}")

# String iteration
print("\nIterating through a string:")
word = "Python"
for char in word:
    print(f"Letter: {char}")

# Multiple sequence iteration using zip
print("\nParallel iteration:")
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old") 