"""
Examples of nested if statements in Python.
"""

def check_login(username, password, is_admin, is_active):
    """Check login credentials with nested conditions."""
    if username and password:  # Check if credentials are provided
        if is_active:  # Check if account is active
            if is_admin:
                print("Welcome, admin!")
                return "admin_dashboard"
            else:
                print("Welcome, user!")
                return "user_dashboard"
        else:
            print("Account is inactive")
            return "activate_account"
    else:
        print("Invalid credentials")
        return "login_page"

def process_payment(amount, balance, is_premium, has_discount):
    """Process payment with multiple conditions."""
    if amount > 0:
        if balance >= amount:
            if is_premium:
                if has_discount:
                    final_amount = amount * 0.8  # 20% discount
                else:
                    final_amount = amount * 0.9  # 10% discount
            else:
                if has_discount:
                    final_amount = amount * 0.9  # 10% discount
                else:
                    final_amount = amount
            
            print(f"Processing payment: ${final_amount:.2f}")
            return True
        else:
            print("Insufficient balance")
            return False
    else:
        print("Invalid amount")
        return False

# Example usage
if __name__ == "__main__":
    # Test login system
    result = check_login("admin", "password123", True, True)
    print(f"Redirect to: {result}")
    
    # Test payment processing
    success = process_payment(100, 150, True, True)
    print(f"Payment successful: {success}") 