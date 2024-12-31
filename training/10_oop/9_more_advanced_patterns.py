"""
Demonstration of more advanced design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy

# Flyweight Pattern
class CharacterStyle:
    """Flyweight class for character styling."""
    
    def __init__(self, font: str, size: int, color: str):
        self.font = font
        self.size = size
        self.color = color

class StyleFactory:
    """Factory for managing character style flyweights."""
    
    _styles: Dict[str, CharacterStyle] = {}
    
    @classmethod
    def get_style(cls, font: str, size: int, color: str) -> CharacterStyle:
        """Get or create a style instance."""
        key = f"{font}-{size}-{color}"
        if key not in cls._styles:
            cls._styles[key] = CharacterStyle(font, size, color)
        return cls._styles[key]
    
    @classmethod
    def style_count(cls) -> int:
        """Get the number of unique styles."""
        return len(cls._styles)

class Character:
    """Context class using flyweight pattern."""
    
    def __init__(self, char: str, style: CharacterStyle):
        self.char = char
        self.style = style
    
    def render(self) -> str:
        """Render the character with its style."""
        return (f"Character '{self.char}' with style: "
                f"{self.style.font}, {self.style.size}pt, {self.style.color}")

# Bridge Pattern
class DrawingAPI(ABC):
    """Implementation interface for drawing."""
    
    @abstractmethod
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        pass
    
    @abstractmethod
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        pass

class DrawingAPI1(DrawingAPI):
    """Concrete implementation 1."""
    
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        print(f"API1: Drawing circle at ({x},{y}), radius {radius}")
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        print(f"API1: Drawing rectangle at ({x},{y}), width {width}, height {height}")

class DrawingAPI2(DrawingAPI):
    """Concrete implementation 2."""
    
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        print(f"API2: Drawing circle at ({x},{y}), radius {radius}")
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        print(f"API2: Drawing rectangle at ({x},{y}), width {width}, height {height}")

class Shape(ABC):
    """Abstraction class."""
    
    def __init__(self, drawing_api: DrawingAPI):
        self._api = drawing_api
    
    @abstractmethod
    def draw(self) -> None:
        pass

class CircleShape(Shape):
    """Refined abstraction for circles."""
    
    def __init__(self, x: int, y: int, radius: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._radius = radius
    
    def draw(self) -> None:
        self._api.draw_circle(self._x, self._y, self._radius)

class RectangleShape(Shape):
    """Refined abstraction for rectangles."""
    
    def __init__(self, x: int, y: int, width: int, height: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._width = width
        self._height = height
    
    def draw(self) -> None:
        self._api.draw_rectangle(self._x, self._y, self._width, self._height)

# Composite Pattern
class FileSystemComponent(ABC):
    """Component interface for file system items."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def display(self, indent: str = "") -> None:
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        pass

class File(FileSystemComponent):
    """Leaf class representing a file."""
    
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self._size = size
    
    def display(self, indent: str = "") -> None:
        print(f"{indent}File: {self.name} ({self._size} bytes)")
    
    def get_size(self) -> int:
        return self._size

class Directory(FileSystemComponent):
    """Composite class representing a directory."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._children: List[FileSystemComponent] = []
    
    def add(self, component: FileSystemComponent) -> None:
        self._children.append(component)
    
    def remove(self, component: FileSystemComponent) -> None:
        self._children.remove(component)
    
    def display(self, indent: str = "") -> None:
        print(f"{indent}Directory: {self.name} ({self.get_size()} bytes)")
        for child in self._children:
            child.display(indent + "  ")
    
    def get_size(self) -> int:
        return sum(child.get_size() for child in self._children)

# Example usage
if __name__ == "__main__":
    # Demonstrate Flyweight Pattern
    print("Flyweight Pattern:")
    factory = StyleFactory()
    text = []
    
    # Create characters with shared styles
    style1 = factory.get_style("Arial", 12, "black")
    style2 = factory.get_style("Times", 14, "red")
    
    text.extend([Character(c, style1) for c in "Hello"])
    text.extend([Character(c, style2) for c in "World"])
    
    for char in text:
        print(char.render())
    print(f"Total unique styles: {factory.style_count()}")
    
    # Demonstrate Bridge Pattern
    print("\nBridge Pattern:")
    api1 = DrawingAPI1()
    api2 = DrawingAPI2()
    
    circle1 = CircleShape(1, 2, 3, api1)
    circle2 = CircleShape(5, 7, 11, api2)
    rect1 = RectangleShape(2, 2, 100, 50, api1)
    
    circle1.draw()
    circle2.draw()
    rect1.draw()
    
    # Demonstrate Composite Pattern
    print("\nComposite Pattern:")
    root = Directory("root")
    
    docs = Directory("docs")
    docs.add(File("resume.pdf", 1024))
    docs.add(File("photo.jpg", 2048))
    
    src = Directory("src")
    src.add(File("main.py", 512))
    src.add(File("utils.py", 256))
    
    root.add(docs)
    root.add(src)
    root.add(File("readme.txt", 128))
    
    root.display() 