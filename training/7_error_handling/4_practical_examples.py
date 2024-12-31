"""
Real-world examples of error handling in Python.
"""

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

def read_configuration(filename):
    """
    Read configuration from a file with error handling.
    """
    config = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip comments and empty lines
                if line.strip() and not line.startswith('#'):
                    try:
                        key, value = line.strip().split('=')
                        config[key.strip()] = value.strip()
                    except ValueError:
                        raise ConfigurationError(f"Invalid configuration line: {line}")
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file '{filename}' not found")
    except PermissionError:
        raise ConfigurationError(f"Permission denied accessing '{filename}'")
    
    return config

def connect_to_database(host, port, timeout=5):
    """
    Simulate database connection with error handling.
    """
    import time
    import random

    try:
        # Simulate network delay
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate potential connection errors
        if random.random() < 0.2:  # 20% chance of timeout
            raise TimeoutError("Connection timed out")
        if random.random() < 0.1:  # 10% chance of refused connection
            raise ConnectionRefusedError("Connection refused")
            
        return {"status": "connected", "host": host, "port": port}
    
    except (TimeoutError, ConnectionRefusedError) as e:
        raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")

def process_user_data(user_input):
    """
    Process user input with comprehensive error handling.
    """
    try:
        # Validate user input
        if not user_input:
            raise ValueError("Input cannot be empty")
        
        # Convert and validate age
        age = int(user_input.get('age', '0'))
        if age <= 0:
            raise ValueError("Age must be a positive number")
        
        # Process email
        email = user_input.get('email', '')
        if not '@' in email:
            raise ValueError("Invalid email format")
        
        return {
            "status": "success",
            "processed_age": age,
            "processed_email": email.lower()
        }
    
    except (ValueError, KeyError) as e:
        return {
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": "An unexpected error occurred",
            "details": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Configuration example
    print("Configuration Example:")
    try:
        config = read_configuration("config.txt")
        print("Configuration loaded successfully:", config)
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    
    # Database connection example
    print("\nDatabase Connection Example:")
    try:
        connection = connect_to_database("localhost", 5432)
        print("Database connected successfully:", connection)
    except DatabaseConnectionError as e:
        print(f"Database error: {e}")
    
    # User data processing example
    print("\nUser Data Processing Example:")
    test_inputs = [
        {"age": "25", "email": "user@example.com"},
        {"age": "-5", "email": "invalid_email"},
        {"age": "abc", "email": "user@example.com"},
        {}
    ]
    
    for input_data in test_inputs:
        result = process_user_data(input_data)
        print(f"\nInput: {input_data}")
        print(f"Result: {result}") 