"""
Demonstration of advanced design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

# Builder Pattern
class Computer:
    """Product class representing a computer."""
    
    def __init__(self):
        self.parts: Dict[str, str] = {}
    
    def add_part(self, part: str, spec: str) -> None:
        self.parts[part] = spec
    
    def show_specs(self) -> None:
        print("\nComputer Specifications:")
        for part, spec in self.parts.items():
            print(f"{part}: {spec}")

class ComputerBuilder(ABC):
    """Abstract builder for computers."""
    
    def __init__(self):
        self.computer = Computer()
    
    @abstractmethod
    def add_cpu(self) -> None:
        pass
    
    @abstractmethod
    def add_memory(self) -> None:
        pass
    
    @abstractmethod
    def add_storage(self) -> None:
        pass
    
    def get_computer(self) -> Computer:
        return self.computer

class GamingComputerBuilder(ComputerBuilder):
    """Concrete builder for gaming computers."""
    
    def add_cpu(self) -> None:
        self.computer.add_part("CPU", "High-end Gaming Processor")
    
    def add_memory(self) -> None:
        self.computer.add_part("Memory", "32GB Gaming RAM")
    
    def add_storage(self) -> None:
        self.computer.add_part("Storage", "2TB Gaming SSD")

class OfficeComputerBuilder(ComputerBuilder):
    """Concrete builder for office computers."""
    
    def add_cpu(self) -> None:
        self.computer.add_part("CPU", "Standard Office Processor")
    
    def add_memory(self) -> None:
        self.computer.add_part("Memory", "8GB Standard RAM")
    
    def add_storage(self) -> None:
        self.computer.add_part("Storage", "256GB SSD")

class ComputerAssembler:
    """Director class that manages the building process."""
    
    def __init__(self, builder: ComputerBuilder):
        self.builder = builder
    
    def construct_computer(self) -> Computer:
        self.builder.add_cpu()
        self.builder.add_memory()
        self.builder.add_storage()
        return self.builder.get_computer()

# Chain of Responsibility Pattern
class Handler(ABC):
    """Abstract handler in chain of responsibility."""
    
    def __init__(self):
        self._next_handler: Optional['Handler'] = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request: str) -> Optional[str]:
        pass

class SpamHandler(Handler):
    """Concrete handler for spam detection."""
    
    def handle(self, request: str) -> Optional[str]:
        if "spam" in request.lower():
            return "Spam detected: Message blocked"
        return self._next_handler.handle(request) if self._next_handler else None

class LengthHandler(Handler):
    """Concrete handler for message length."""
    
    def handle(self, request: str) -> Optional[str]:
        if len(request) > 100:
            return "Message too long: Please shorten"
        return self._next_handler.handle(request) if self._next_handler else None

class ProfanityHandler(Handler):
    """Concrete handler for profanity."""
    
    def handle(self, request: str) -> Optional[str]:
        profanity = ["bad", "worst", "terrible"]
        if any(word in request.lower() for word in profanity):
            return "Profanity detected: Message blocked"
        return self._next_handler.handle(request) if self._next_handler else None

# Mediator Pattern
class ChatMediator:
    """Mediator for chat communication."""
    
    def __init__(self):
        self._users: List['ChatUser'] = []
    
    def add_user(self, user: 'ChatUser') -> None:
        self._users.append(user)
    
    def send_message(self, message: str, sender: 'ChatUser') -> None:
        for user in self._users:
            if user != sender:
                user.receive(message, sender.name)

class ChatUser:
    """Colleague class for chat users."""
    
    def __init__(self, name: str, mediator: ChatMediator):
        self.name = name
        self._mediator = mediator
        mediator.add_user(self)
    
    def send(self, message: str) -> None:
        print(f"{self.name} sends: {message}")
        self._mediator.send_message(message, self)
    
    def receive(self, message: str, sender_name: str) -> None:
        print(f"{self.name} receives from {sender_name}: {message}")

# Example usage
if __name__ == "__main__":
    # Demonstrate Builder Pattern
    print("Builder Pattern:")
    
    # Build a gaming computer
    gaming_builder = GamingComputerBuilder()
    assembler = ComputerAssembler(gaming_builder)
    gaming_pc = assembler.construct_computer()
    gaming_pc.show_specs()
    
    # Build an office computer
    office_builder = OfficeComputerBuilder()
    assembler = ComputerAssembler(office_builder)
    office_pc = assembler.construct_computer()
    office_pc.show_specs()
    
    # Demonstrate Chain of Responsibility
    print("\nChain of Responsibility Pattern:")
    spam_handler = SpamHandler()
    length_handler = LengthHandler()
    profanity_handler = ProfanityHandler()
    
    spam_handler.set_next(length_handler).set_next(profanity_handler)
    
    messages = [
        "Hello, this is a normal message",
        "SPAM SPAM SPAM",
        "This message is way too long " * 10,
        "This product is terrible"
    ]
    
    for msg in messages:
        result = spam_handler.handle(msg)
        if result:
            print(f"Message: '{msg[:50]}...' -> {result}")
        else:
            print(f"Message: '{msg[:50]}...' -> Accepted")
    
    # Demonstrate Mediator Pattern
    print("\nMediator Pattern:")
    mediator = ChatMediator()
    
    alice = ChatUser("Alice", mediator)
    bob = ChatUser("Bob", mediator)
    charlie = ChatUser("Charlie", mediator)
    
    alice.send("Hello everyone!")
    bob.send("Hi Alice!")
    charlie.send("Hey team!") 