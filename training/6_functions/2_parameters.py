"""
Demonstration of different parameter types and argument passing in Python functions.
"""

# Required parameters
def greet(name, greeting):
    """Function with required parameters."""
    print(f"{greeting}, {name}!")

# Default parameters
def create_profile(name, age=25, city="Unknown"):
    """Function with default parameters."""
    return {
        "name": name,
        "age": age,
        "city": city
    }

# Variable number of arguments
def calculate_average(*numbers):
    """Function accepting variable number of arguments."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# Keyword arguments
def format_name(first, last, middle=""):
    """Function demonstrating keyword arguments."""
    if middle:
        return f"{first} {middle} {last}"
    return f"{first} {last}"

# Mixing positional and keyword arguments
def create_email(username, domain="example.com", **headers):
    """Function with mixed argument types."""
    email = f"{username}@{domain}"
    if headers:
        return {"email": email, "headers": headers}
    return {"email": email}

# Keyword-only arguments
def configure_settings(*, theme="light", language="en", notifications=True):
    """Function with keyword-only arguments."""
    return {
        "theme": theme,
        "language": language,
        "notifications": notifications
    }

# Example usage
if __name__ == "__main__":
    # Required parameters
    print("Required parameters:")
    greet("Alice", "Good morning")
    
    # Default parameters
    print("\nDefault parameters:")
    profile1 = create_profile("Bob")
    profile2 = create_profile("Charlie", age=30, city="New York")
    print(f"Profile 1: {profile1}")
    print(f"Profile 2: {profile2}")
    
    # Variable arguments
    print("\nVariable arguments:")
    avg1 = calculate_average(1, 2, 3, 4, 5)
    avg2 = calculate_average(10, 20)
    print(f"Average 1: {avg1}")
    print(f"Average 2: {avg2}")
    
    # Keyword arguments
    print("\nKeyword arguments:")
    name1 = format_name("John", "Doe")
    name2 = format_name("Jane", "Smith", middle="Marie")
    print(f"Name 1: {name1}")
    print(f"Name 2: {name2}")
    
    # Mixed arguments
    print("\nMixed arguments:")
    email1 = create_email("john.doe")
    email2 = create_email("jane.smith", "company.com", reply_to="support@company.com")
    print(f"Email 1: {email1}")
    print(f"Email 2: {email2}")
    
    # Keyword-only arguments
    print("\nKeyword-only arguments:")
    settings = configure_settings(theme="dark", notifications=False)
    print(f"Settings: {settings}") 