"""
Examples of loop control statements: break, continue, and else.
"""

# Break example - Finding first even number
print("Break Example:")
numbers = [1, 3, 5, 6, 7, 8, 9]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number found: {num}")
        break

# Continue example - Skipping even numbers
print("\nContinue Example:")
for num in range(1, 6):
    if num % 2 == 0:
        continue
    print(f"Odd number: {num}")

# Loop with else - Search example
print("\nLoop with else:")
search_term = "python"
languages = ["java", "javascript", "ruby"]

for lang in languages:
    if lang == search_term:
        print(f"Found {search_term}!")
        break
else:
    print(f"Could not find {search_term}")

# Nested loop with break
print("\nNested Loop Break:")
for i in range(3):
    for j in range(3):
        if i == j == 1:
            print("Breaking inner loop")
            break
        print(f"Position: ({i}, {j})")

# Continue with counter
print("\nContinue with Counter:")
skipped = 0
for i in range(10):
    if i % 3 == 0:
        skipped += 1
        continue
    print(f"Processing number: {i}")
print(f"Skipped {skipped} numbers") 