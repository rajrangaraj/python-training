"""
Demonstration of abstract classes and interfaces in Python.
"""

from abc import ABC, abstractmethod
from typing import Protocol

# Abstract Base Class example
class Shape(ABC):
    """Abstract base class for shapes."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        """Calculate area of the shape."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter of the shape."""
        pass
    
    def describe(self):
        """Describe the shape with its measurements."""
        return (f"{self.name} - "
                f"Area: {self.area():.2f}, "
                f"Perimeter: {self.perimeter():.2f}")

class Circle(Shape):
    """Concrete implementation of a circle."""
    
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        """Calculate circle area."""
        from math import pi
        return pi * self.radius ** 2
    
    def perimeter(self):
        """Calculate circle circumference."""
        from math import pi
        return 2 * pi * self.radius

class Rectangle(Shape):
    """Concrete implementation of a rectangle."""
    
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        """Calculate rectangle area."""
        return self.width * self.height
    
    def perimeter(self):
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)

# Protocol (Interface) example
class Drawable(Protocol):
    """Protocol defining drawable objects."""
    
    def draw(self) -> str:
        """Draw the object."""
        ...

class Movable(Protocol):
    """Protocol defining movable objects."""
    
    def move(self, x: float, y: float) -> None:
        """Move the object."""
        ...

class GraphicalShape(Shape):
    """Shape that can be drawn and moved."""
    
    def __init__(self, name, x=0, y=0):
        super().__init__(name)
        self.x = x
        self.y = y
    
    def draw(self) -> str:
        """Draw the shape."""
        return f"Drawing {self.name} at ({self.x}, {self.y})"
    
    def move(self, x: float, y: float) -> None:
        """Move the shape."""
        self.x = x
        self.y = y

class GraphicalCircle(GraphicalShape, Circle):
    """Circle that can be drawn and moved."""
    
    def __init__(self, radius, x=0, y=0):
        Circle.__init__(self, radius)
        GraphicalShape.__init__(self, "Circle", x, y)
    
    def draw(self) -> str:
        """Draw the circle."""
        return (f"Drawing Circle with radius {self.radius} "
                f"at ({self.x}, {self.y})")

def draw_shape(drawable: Drawable):
    """Function that works with any drawable object."""
    print(drawable.draw())

def move_shape(movable: Movable, x: float, y: float):
    """Function that works with any movable object."""
    movable.move(x, y)
    print(f"Moved to ({x}, {y})")

# Example usage
if __name__ == "__main__":
    # Create shapes
    circle = Circle(5)
    rectangle = Rectangle(4, 6)
    
    # Use abstract methods
    print("Basic Shapes:")
    print(circle.describe())
    print(rectangle.describe())
    
    # Try to instantiate abstract class
    try:
        shape = Shape("Generic")  # This will raise TypeError
    except TypeError as e:
        print(f"\nCannot instantiate abstract class: {e}")
    
    # Use drawable and movable shapes
    g_circle = GraphicalCircle(3, 1, 1)
    print("\nGraphical Shape:")
    print(g_circle.describe())
    draw_shape(g_circle)
    move_shape(g_circle, 5, 5)
    draw_shape(g_circle)
    
    # Demonstrate isinstance checks
    print("\nType checking:")
    print(f"Is circle a Shape? {isinstance(circle, Shape)}")
    print(f"Is g_circle a Circle? {isinstance(g_circle, Circle)}")
    print(f"Is g_circle Drawable? {isinstance(g_circle, Drawable)}")
    print(f"Is g_circle Movable? {isinstance(g_circle, Movable)}") 