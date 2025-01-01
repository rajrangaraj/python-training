"""
Demonstration of SOLID principles and clean architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Protocol, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Single Responsibility Principle (SRP)
# Each class has only one reason to change

class UserRepository:
    """Handles user data storage operations."""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
    
    def save(self, user: Dict[str, Any]):
        """Save user data to storage."""
        with open(self.database_path, 'a') as f:
            json.dump(user, f)
            f.write('\n')
    
    def find_by_id(self, user_id: str) -> Dict[str, Any]:
        """Find user by ID."""
        with open(self.database_path, 'r') as f:
            for line in f:
                user = json.loads(line)
                if user['id'] == user_id:
                    return user
        return None

class UserValidator:
    """Handles user data validation."""
    
    def validate(self, user: Dict[str, Any]) -> List[str]:
        errors = []
        
        if not user.get('email'):
            errors.append("Email is required")
        elif '@' not in user['email']:
            errors.append("Invalid email format")
        
        if not user.get('name'):
            errors.append("Name is required")
        
        return errors

# Open/Closed Principle (OCP)
# Open for extension, closed for modification

class PaymentMethod(Protocol):
    def process_payment(self, amount: float) -> bool:
        ...

class CreditCardPayment:
    def process_payment(self, amount: float) -> bool:
        print(f"Processing ${amount} via Credit Card")
        return True

class PayPalPayment:
    def process_payment(self, amount: float) -> bool:
        print(f"Processing ${amount} via PayPal")
        return True

class PaymentProcessor:
    def __init__(self, payment_method: PaymentMethod):
        self.payment_method = payment_method
    
    def process(self, amount: float) -> bool:
        return self.payment_method.process_payment(amount)

# Liskov Substitution Principle (LSP)
# Subtypes must be substitutable for their base types

@dataclass
class Rectangle:
    width: float
    height: float
    
    def area(self) -> float:
        return self.width * self.height

@dataclass
class Square(Rectangle):
    size: float
    
    def __post_init__(self):
        self.width = self.size
        self.height = self.size

# Interface Segregation Principle (ISP)
# Clients shouldn't depend on interfaces they don't use

class Printer(Protocol):
    def print_document(self, document: str):
        ...

class Scanner(Protocol):
    def scan_document(self) -> str:
        ...

class Fax(Protocol):
    def fax_document(self, document: str, recipient: str):
        ...

class SimplePrinter:
    def print_document(self, document: str):
        print(f"Printing: {document}")

class AllInOnePrinter:
    def print_document(self, document: str):
        print(f"Printing: {document}")
    
    def scan_document(self) -> str:
        return "Scanned document content"
    
    def fax_document(self, document: str, recipient: str):
        print(f"Faxing {document} to {recipient}")

# Dependency Inversion Principle (DIP)
# High-level modules shouldn't depend on low-level modules

class EmailService(Protocol):
    def send_email(self, to: str, subject: str, body: str):
        ...

class SMTPEmailService:
    def send_email(self, to: str, subject: str, body: str):
        print(f"Sending email to {to}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")

class NotificationService:
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
    
    def notify_user(self, user_email: str, message: str):
        self.email_service.send_email(
            to=user_email,
            subject="Notification",
            body=message
        )

# Clean Architecture Example

class UserInputPort(Protocol):
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        ...

class UserOutputPort(Protocol):
    def save_user(self, user: Dict[str, Any]):
        ...
    def user_exists(self, email: str) -> bool:
        ...

class UserUseCase:
    def __init__(self, user_repository: UserOutputPort):
        self.user_repository = user_repository
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.user_repository.user_exists(user_data['email']):
            raise ValueError("User already exists")
        
        user = {
            'id': str(hash(user_data['email'])),
            'created_at': datetime.now().isoformat(),
            **user_data
        }
        
        self.user_repository.save_user(user)
        return user

def demonstrate_principles():
    """Demonstrate SOLID principles and clean architecture."""
    
    print("\n1. Single Responsibility Principle:")
    repository = UserRepository("users.json")
    validator = UserValidator()
    
    user_data = {
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    errors = validator.validate(user_data)
    if not errors:
        repository.save(user_data)
    
    print("\n2. Open/Closed Principle:")
    processor = PaymentProcessor(CreditCardPayment())
    processor.process(100.0)
    
    processor = PaymentProcessor(PayPalPayment())
    processor.process(50.0)
    
    print("\n3. Liskov Substitution Principle:")
    shapes: List[Rectangle] = [
        Rectangle(width=4, height=5),
        Square(size=4)
    ]
    for shape in shapes:
        print(f"Area: {shape.area()}")
    
    print("\n4. Interface Segregation Principle:")
    simple_printer = SimplePrinter()
    all_in_one = AllInOnePrinter()
    
    simple_printer.print_document("Test document")
    all_in_one.print_document("Test document")
    all_in_one.scan_document()
    
    print("\n5. Dependency Inversion Principle:")
    email_service = SMTPEmailService()
    notification_service = NotificationService(email_service)
    notification_service.notify_user("user@example.com", "Hello!")
    
    print("\n6. Clean Architecture:")
    user_repository = UserRepository("users.json")
    user_use_case = UserUseCase(user_repository)
    
    try:
        new_user = user_use_case.create_user({
            "name": "Jane Doe",
            "email": "jane@example.com"
        })
        print(f"Created user: {new_user}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    demonstrate_principles() 