"""
Examples of logical operators (and, or, not) in Python.
"""

def check_eligibility(age, income, credit_score):
    """Check loan eligibility using logical operators."""
    # All conditions must be met (AND)
    if age >= 18 and income >= 30000 and credit_score >= 700:
        print("Fully eligible for loan")
        return True
    # At least some conditions met (OR)
    elif age >= 18 or income >= 50000 or credit_score >= 800:
        print("Partially eligible, needs review")
        return "REVIEW"
    else:
        print("Not eligible")
        return False

def validate_password(password):
    """Validate password using multiple logical conditions."""
    has_length = len(password) >= 8
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)

    # Using AND to check all conditions
    if has_length and has_upper and has_lower and has_digit and has_special:
        return "Strong password"
    # Using OR to check partial conditions
    elif has_length and ((has_upper and has_lower) or (has_digit and has_special)):
        return "Moderate password"
    else:
        return "Weak password"

def check_access(user_role, is_logged_in, is_admin):
    """Check access permissions using logical operators."""
    # Using NOT operator
    if not is_logged_in:
        return "Please log in first"
    
    # Complex logical combinations
    if is_admin or (user_role == "manager" and is_logged_in):
        return "Full access granted"
    elif user_role in ["user", "guest"] and is_logged_in and not is_admin:
        return "Limited access granted"
    else:
        return "Access denied"

# Example usage
if __name__ == "__main__":
    # Test loan eligibility
    print("\nLoan Eligibility Test:")
    result = check_eligibility(25, 45000, 720)
    print(f"Application status: {result}")
    
    # Test password strength
    print("\nPassword Strength Test:")
    passwords = ["abc123", "Password123!", "weakpass"]
    for pwd in passwords:
        strength = validate_password(pwd)
        print(f"Password '{pwd}': {strength}")
    
    # Test access control
    print("\nAccess Control Test:")
    access = check_access("manager", True, False)
    print(f"Access level: {access}") 