"""
Real-world practical examples using Python functions.
"""

def calculate_total_cost(items, tax_rate=0.1, shipping_fee=5.0):
    """
    Calculate total cost including tax and shipping.
    
    Args:
        items: List of (item, price) tuples
        tax_rate: Tax rate as decimal (default 0.1 = 10%)
        shipping_fee: Flat shipping fee (default $5.00)
    """
    subtotal = sum(price for _, price in items)
    tax = subtotal * tax_rate
    total = subtotal + tax + shipping_fee
    
    return {
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping_fee,
        "total": total,
        "items_count": len(items)
    }

def generate_password(length=12, include_special=True):
    """Generate a random password."""
    import random
    import string
    
    # Define character sets
    letters = string.ascii_letters
    digits = string.digits
    special = string.punctuation if include_special else ""
    
    # Ensure at least one of each required type
    password = [
        random.choice(string.ascii_lowercase),
        random.choice(string.ascii_uppercase),
        random.choice(string.digits)
    ]
    
    if include_special:
        password.append(random.choice(special))
    
    # Fill the rest randomly
    all_chars = letters + digits + special
    password.extend(random.choice(all_chars) 
                   for _ in range(length - len(password)))
    
    # Shuffle the password
    random.shuffle(password)
    return "".join(password)

def analyze_student_grades(grades_dict):
    """Analyze student grades and return statistics."""
    if not grades_dict:
        return None
        
    grades = list(grades_dict.values())
    
    return {
        "highest": max(grades),
        "lowest": min(grades),
        "average": sum(grades) / len(grades),
        "passing": sum(1 for grade in grades if grade >= 60),
        "failing": sum(1 for grade in grades if grade < 60),
        "total_students": len(grades)
    }

# Example usage
if __name__ == "__main__":
    # Test shopping cart
    print("Shopping Cart Example:")
    cart_items = [
        ("Book", 29.99),
        ("Shirt", 19.99),
        ("Headphones", 99.99)
    ]
    result = calculate_total_cost(cart_items)
    for key, value in result.items():
        print(f"{key}: ${value:.2f}")
    
    # Test password generator
    print("\nPassword Generator Example:")
    for _ in range(3):
        pwd = generate_password()
        print(f"Generated password: {pwd}")
    
    # Test grade analysis
    print("\nGrade Analysis Example:")
    grades = {
        "Alice": 92,
        "Bob": 85,
        "Charlie": 78,
        "David": 55,
        "Eve": 98
    }
    analysis = analyze_student_grades(grades)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}") 