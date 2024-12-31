"""
Demonstration of composition and mixins in Python.
"""

from dataclasses import dataclass
from typing import List, Optional

# Component classes for composition
@dataclass
class Address:
    """Component class for addresses."""
    street: str
    city: str
    state: str
    zip_code: str
    
    def __str__(self):
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"

@dataclass
class ContactInfo:
    """Component class for contact information."""
    email: str
    phone: Optional[str] = None
    
    def __str__(self):
        if self.phone:
            return f"Email: {self.email}, Phone: {self.phone}"
        return f"Email: {self.email}"

# Mixin classes
class LoggerMixin:
    """Mixin to add logging capabilities."""
    
    def log(self, message: str) -> None:
        """Log a message with the class name."""
        print(f"[{self.__class__.__name__}] {message}")

class TimestampMixin:
    """Mixin to add timestamp capabilities."""
    
    def __init__(self):
        from datetime import datetime
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        from datetime import datetime
        self.updated_at = datetime.now()

# Main classes using composition and mixins
class Person(LoggerMixin, TimestampMixin):
    """Class demonstrating composition and mixins."""
    
    def __init__(self, name: str, address: Address, contact: ContactInfo):
        super().__init__()  # Initialize TimestampMixin
        self.name = name
        self.address = address
        self.contact = contact
        self.log(f"Created new person: {name}")
    
    def update_address(self, new_address: Address):
        """Update person's address."""
        self.address = new_address
        self.update_timestamp()
        self.log(f"Updated address for {self.name}")
    
    def update_contact(self, new_contact: ContactInfo):
        """Update person's contact information."""
        self.contact = new_contact
        self.update_timestamp()
        self.log(f"Updated contact info for {self.name}")
    
    def __str__(self):
        return (f"Name: {self.name}\n"
                f"Address: {self.address}\n"
                f"Contact: {self.contact}")

class Department(LoggerMixin):
    """Class demonstrating composition with multiple objects."""
    
    def __init__(self, name: str):
        self.name = name
        self.employees: List[Person] = []
        self.log(f"Created new department: {name}")
    
    def add_employee(self, employee: Person):
        """Add an employee to the department."""
        self.employees.append(employee)
        self.log(f"Added {employee.name} to {self.name} department")
    
    def remove_employee(self, employee: Person):
        """Remove an employee from the department."""
        if employee in self.employees:
            self.employees.remove(employee)
            self.log(f"Removed {employee.name} from {self.name} department")
    
    def list_employees(self):
        """List all employees in the department."""
        print(f"\nEmployees in {self.name} department:")
        for employee in self.employees:
            print(f"- {employee.name}")

# Example usage
if __name__ == "__main__":
    # Create component objects
    home_address = Address("123 Main St", "Anytown", "CA", "12345")
    work_address = Address("456 Corp Ave", "Business City", "CA", "67890")
    
    contact1 = ContactInfo("john@example.com", "555-0123")
    contact2 = ContactInfo("jane@example.com")
    
    # Create persons using composition
    john = Person("John Doe", home_address, contact1)
    jane = Person("Jane Smith", work_address, contact2)
    
    # Demonstrate logging mixin
    print("\nCreation logs were automatically generated:")
    
    # Demonstrate timestamp mixin
    print(f"\nJohn's record created at: {john.created_at}")
    
    # Update information
    new_address = Address("789 New St", "Newtown", "CA", "54321")
    john.update_address(new_address)
    
    print(f"John's record updated at: {john.updated_at}")
    
    # Create and populate department
    engineering = Department("Engineering")
    engineering.add_employee(john)
    engineering.add_employee(jane)
    
    # List department members
    engineering.list_employees()
    
    # Display full information
    print("\nDetailed Information:")
    print("\nJohn's Info:")
    print(john)
    print("\nJane's Info:")
    print(jane) 