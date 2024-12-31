"""
Creating and using custom exceptions in Python.
"""

# Custom exception classes
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class InsufficientFundsError(Exception):
    """Custom exception for insufficient funds."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        self.message = f"Insufficient funds: balance=${balance}, required=${amount}"
        super().__init__(self.message)

class AgeRestrictionError(Exception):
    """Custom exception for age restrictions."""
    def __init__(self, min_age, actual_age):
        self.min_age = min_age
        self.actual_age = actual_age
        super().__init__(f"Must be at least {min_age} years old (not {actual_age})")

# Functions using custom exceptions
def validate_username(username):
    """Validate username with custom exception."""
    if len(username) < 3:
        raise ValidationError("Username must be at least 3 characters long")
    if not username.isalnum():
        raise ValidationError("Username must contain only letters and numbers")
    return True

def process_payment(balance, amount):
    """Process payment with custom exception."""
    if amount <= 0:
        raise ValidationError("Amount must be positive")
    if balance < amount:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

def verify_age(age, required_age=18):
    """Verify age with custom exception."""
    if age < required_age:
        raise AgeRestrictionError(required_age, age)
    return True

# Example usage
if __name__ == "__main__":
    # Username validation
    print("Username Validation:")
    usernames = ["ab", "user123", "user@123"]
    for username in usernames:
        try:
            validate_username(username)
            print(f"Username '{username}' is valid")
        except ValidationError as e:
            print(f"Invalid username '{username}': {str(e)}")
    
    # Payment processing
    print("\nPayment Processing:")
    try:
        balance = 100
        amount = 150
        process_payment(balance, amount)
    except InsufficientFundsError as e:
        print(e.message)
    
    # Age verification
    print("\nAge Verification:")
    try:
        verify_age(16, 21)
    except AgeRestrictionError as e:
        print(f"Access denied: {str(e)}") 