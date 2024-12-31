"""
Custom module for geometric calculations.
"""

import math

def calculate_circle_area(radius):
    """Calculate the area of a circle."""
    return math.pi * radius ** 2

def calculate_rectangle_area(length, width):
    """Calculate the area of a rectangle."""
    return length * width

def calculate_triangle_area(base, height):
    """Calculate the area of a triangle."""
    return 0.5 * base * height

class Shape:
    """Base class for geometric shapes."""
    
    def __init__(self, name):
        self.name = name
    
    def get_area(self):
        """To be implemented by subclasses."""
        raise NotImplementedError
    
    def get_name(self):
        """Return the shape name."""
        return self.name

class Circle(Shape):
    """Circle shape class."""
    
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def get_area(self):
        """Calculate circle area."""
        return calculate_circle_area(self.radius)
    
    def get_circumference(self):
        """Calculate circle circumference."""
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    """Rectangle shape class."""
    
    def __init__(self, length, width):
        super().__init__("Rectangle")
        self.length = length
        self.width = width
    
    def get_area(self):
        """Calculate rectangle area."""
        return calculate_rectangle_area(self.length, self.width)
    
    def get_perimeter(self):
        """Calculate rectangle perimeter."""
        return 2 * (self.length + self.width)

# Example usage
if __name__ == "__main__":
    # Test basic functions
    print(f"Circle area (r=5): {calculate_circle_area(5):.2f}")
    print(f"Rectangle area (4x6): {calculate_rectangle_area(4, 6)}")
    print(f"Triangle area (b=3, h=4): {calculate_triangle_area(3, 4)}")
    
    # Test shape classes
    circle = Circle(5)
    rectangle = Rectangle(4, 6)
    
    print(f"\n{circle.get_name()} area: {circle.get_area():.2f}")
    print(f"{circle.get_name()} circumference: {circle.get_circumference():.2f}")
    
    print(f"\n{rectangle.get_name()} area: {rectangle.get_area()}")
    print(f"{rectangle.get_name()} perimeter: {rectangle.get_perimeter()}") 