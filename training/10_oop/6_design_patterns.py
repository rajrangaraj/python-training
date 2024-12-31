"""
Demonstration of common design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime

# Singleton Pattern
class Logger:
    """Singleton logger class."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.log_history = []
        return cls._instance
    
    def log(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.log_history.append(entry)
        print(entry)
    
    def get_history(self) -> List[str]:
        """Get the log history."""
        return self.log_history.copy()

# Factory Pattern
class Document(ABC):
    """Abstract document class."""
    
    @abstractmethod
    def create(self) -> str:
        pass

class PDFDocument(Document):
    """Concrete PDF document."""
    
    def create(self) -> str:
        return "Created PDF document"

class WordDocument(Document):
    """Concrete Word document."""
    
    def create(self) -> str:
        return "Created Word document"

class DocumentFactory:
    """Factory for creating documents."""
    
    @staticmethod
    def create_document(doc_type: str) -> Document:
        """Create a document of the specified type."""
        if doc_type.lower() == "pdf":
            return PDFDocument()
        elif doc_type.lower() == "word":
            return WordDocument()
        raise ValueError(f"Unknown document type: {doc_type}")

# Observer Pattern
class Subject(ABC):
    """Abstract subject class."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: 'Observer') -> None:
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: 'Observer') -> None:
        """Detach an observer."""
        self._observers.remove(observer)
    
    def notify(self, data: Any) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(data)

class Observer(ABC):
    """Abstract observer class."""
    
    @abstractmethod
    def update(self, data: Any) -> None:
        pass

class NewsAgency(Subject):
    """Concrete subject class."""
    
    def publish_news(self, news: str) -> None:
        """Publish news to all observers."""
        self.notify(news)

class NewsSubscriber(Observer):
    """Concrete observer class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def update(self, news: str) -> None:
        """Receive news updates."""
        print(f"{self.name} received news: {news}")

# Command Pattern
class Command(ABC):
    """Abstract command class."""
    
    @abstractmethod
    def execute(self) -> None:
        pass
    
    @abstractmethod
    def undo(self) -> None:
        pass

class TextEditor:
    """Text editor class."""
    
    def __init__(self):
        self.text = ""
    
    def write(self, text: str) -> None:
        """Write text."""
        self.text += text
    
    def delete(self, length: int) -> None:
        """Delete last n characters."""
        self.text = self.text[:-length]
    
    def get_text(self) -> str:
        """Get current text."""
        return self.text

class WriteCommand(Command):
    """Concrete write command."""
    
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text
    
    def execute(self) -> None:
        self.editor.write(self.text)
    
    def undo(self) -> None:
        self.editor.delete(len(self.text))

# Example usage
if __name__ == "__main__":
    # Demonstrate Singleton
    print("Singleton Pattern:")
    logger1 = Logger()
    logger2 = Logger()
    print(f"Same logger instance: {logger1 is logger2}")
    logger1.log("Test message")
    
    # Demonstrate Factory
    print("\nFactory Pattern:")
    factory = DocumentFactory()
    documents = [
        factory.create_document("PDF"),
        factory.create_document("Word")
    ]
    for doc in documents:
        print(doc.create())
    
    # Demonstrate Observer
    print("\nObserver Pattern:")
    agency = NewsAgency()
    subscriber1 = NewsSubscriber("John")
    subscriber2 = NewsSubscriber("Jane")
    
    agency.attach(subscriber1)
    agency.attach(subscriber2)
    agency.publish_news("Breaking news!")
    
    agency.detach(subscriber1)
    agency.publish_news("Another update!")
    
    # Demonstrate Command
    print("\nCommand Pattern:")
    editor = TextEditor()
    commands: List[Command] = [
        WriteCommand(editor, "Hello "),
        WriteCommand(editor, "World!")
    ]
    
    # Execute commands
    for cmd in commands:
        cmd.execute()
    print(f"Current text: {editor.get_text()}")
    
    # Undo commands
    for cmd in reversed(commands):
        cmd.undo()
    print(f"After undo: {editor.get_text()}") 