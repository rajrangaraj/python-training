"""
Demonstration of variable scope and namespace concepts in Python functions.
"""

# Global variable
message = "Global message"

def demonstrate_scope():
    """Show different scope levels."""
    # Local variable
    local_message = "Local message"
    print(f"Inside function - Local: {local_message}")
    print(f"Inside function - Global: {message}")

def modify_global():
    """Demonstrate global keyword."""
    global message
    message = "Modified global message"
    print(f"Inside function - Modified global: {message}")

def nested_scope():
    """Demonstrate nested scope."""
    outer_value = "Outer value"
    
    def inner_function():
        inner_value = "Inner value"
        print(f"Inner function - Local: {inner_value}")
        print(f"Inner function - Outer: {outer_value}")
        print(f"Inner function - Global: {message}")
    
    inner_function()
    print(f"Outer function - Local: {outer_value}")

def demonstrate_nonlocal():
    """Demonstrate nonlocal keyword."""
    count = 0
    
    def increment():
        nonlocal count
        count += 1
        print(f"Count inside increment(): {count}")
    
    increment()
    print(f"Count after increment(): {count}")

# Example usage
if __name__ == "__main__":
    print("Basic scope example:")
    demonstrate_scope()
    
    print("\nModifying global variable:")
    print(f"Before modification: {message}")
    modify_global()
    print(f"After modification: {message}")
    
    print("\nNested scope example:")
    nested_scope()
    
    print("\nNonlocal example:")
    demonstrate_nonlocal()