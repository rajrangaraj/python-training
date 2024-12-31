"""
Demonstration of Python's exception hierarchy and exception chaining.
"""

def demonstrate_exception_hierarchy():
    """Show different types of built-in exceptions."""
    examples = [
        # TypeError example
        lambda: len(42),
        
        # ValueError example
        lambda: int("abc"),
        
        # IndexError example
        lambda: [1, 2, 3][10],
        
        # KeyError example
        lambda: {"a": 1}["b"],
        
        # AttributeError example
        lambda: "string".undefined_method(),
        
        # ZeroDivisionError example
        lambda: 1/0
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nExample {i}:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print(f"Exception hierarchy: {' -> '.join(str(c.__name__) for c in type(e).__mro__)}")

def demonstrate_exception_chaining():
    """Show exception chaining with 'raise from'."""
    try:
        try:
            # Original error
            result = int("abc")
        except ValueError as e:
            # Chain a custom exception
            raise RuntimeError("Failed to process input") from e
    except RuntimeError as e:
        print("\nException Chaining Example:")
        print(f"Current exception: {type(e).__name__}: {str(e)}")
        print(f"Original exception: {type(e.__cause__).__name__}: {str(e.__cause__)}")

def demonstrate_exception_groups():
    """Demonstrate handling multiple exceptions together (Python 3.11+)."""
    def process_items(items):
        errors = []
        results = []
        
        for i, item in enumerate(items):
            try:
                if isinstance(item, str):
                    results.append(float(item))
                else:
                    results.append(item / 2)
            except Exception as e:
                errors.append((i, e))
        
        if errors:
            error_messages = [f"Item {i}: {str(e)}" for i, e in errors]
            raise Exception("Multiple errors occurred:\n" + "\n".join(error_messages))
        
        return results

    # Test with problematic data
    test_data = ["1.5", "abc", 10, 0, "2.5"]
    try:
        results = process_items(test_data)
        print("\nAll items processed successfully:", results)
    except Exception as e:
        print("\nError processing items:")
        print(str(e))

# Example usage
if __name__ == "__main__":
    print("Exception Hierarchy Examples:")
    demonstrate_exception_hierarchy()
    
    print("\nException Chaining Example:")
    demonstrate_exception_chaining()
    
    print("\nException Groups Example:")
    demonstrate_exception_groups() 