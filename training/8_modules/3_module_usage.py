"""
Examples of using both built-in and custom modules.
"""

# Import custom modules
from calculator import Calculator, add, PI
from geometry import Circle, Rectangle, calculate_triangle_area

# Import built-in modules
import json
import os
from datetime import datetime

def demonstrate_calculator():
    """Show usage of calculator module."""
    print("Calculator Module Examples:")
    
    # Use individual functions
    print(f"Direct addition: {add(5, 3)}")
    print(f"Using PI constant: {PI}")
    
    # Use calculator class
    calc = Calculator()
    operations = [
        ('add', 10, 5),
        ('multiply', 4, 3),
        ('divide', 15, 3)
    ]
    
    for op, a, b in operations:
        result = calc.calculate(op, a, b)
        print(f"{op}: {a}, {b} = {result}")
    
    print("Calculation history:", calc.get_history())

def demonstrate_geometry():
    """Show usage of geometry module."""
    print("\nGeometry Module Examples:")
    
    # Create shapes
    circle = Circle(radius=5)
    rectangle = Rectangle(length=4, width=6)
    
    # Calculate and display areas
    print(f"Circle area: {circle.get_area():.2f}")
    print(f"Rectangle area: {rectangle.get_area()}")
    print(f"Triangle area: {calculate_triangle_area(3, 4)}")

def demonstrate_built_in_modules():
    """Show usage of built-in modules."""
    print("\nBuilt-in Modules Examples:")
    
    # JSON module
    data = {
        "name": "John",
        "age": 30,
        "city": "New York"
    }
    json_string = json.dumps(data, indent=2)
    print(f"JSON output:\n{json_string}")
    
    # OS module
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Datetime module
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCurrent time: {formatted_date}")

def save_results(filename):
    """Save calculation results to file."""
    calc = Calculator()
    results = []
    
    # Perform some calculations
    operations = [
        ('add', 10, 5),
        ('multiply', 4, 3),
        ('divide', 15, 3)
    ]
    
    for op, a, b in operations:
        result = calc.calculate(op, a, b)
        results.append({
            "operation": op,
            "a": a,
            "b": b,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

# Example usage
if __name__ == "__main__":
    demonstrate_calculator()
    demonstrate_geometry()
    demonstrate_built_in_modules()
    save_results("calculation_results.json") 