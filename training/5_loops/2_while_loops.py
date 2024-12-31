"""
Detailed examples of while loops and their applications.
"""

# Basic countdown
print("Countdown:")
count = 5
while count > 0:
    print(count)
    count -= 1
print("Blast off!")

# Input validation
print("\nInput Validation:")
while True:
    try:
        age = int(input("Enter your age (1-120): "))
        if 1 <= age <= 120:
            print(f"Valid age: {age}")
            break
        else:
            print("Age must be between 1 and 120")
    except ValueError:
        print("Please enter a valid number")

# Password attempt limiting
print("\nPassword Check:")
correct_password = "secret123"
attempts = 3
while attempts > 0:
    password = input(f"Enter password ({attempts} attempts left): ")
    if password == correct_password:
        print("Access granted!")
        break
    attempts -= 1
    if attempts > 0:
        print("Incorrect password, try again")
else:
    print("Account locked - too many attempts")

# Number guessing game
import random
print("\nGuessing Game:")
target = random.randint(1, 10)
guesses = 0
while True:
    guesses += 1
    try:
        guess = int(input("Guess the number (1-10): "))
        if guess == target:
            print(f"Correct! It took you {guesses} guesses")
            break
        elif guess < target:
            print("Too low!")
        else:
            print("Too high!")
    except ValueError:
        print("Please enter a valid number") 