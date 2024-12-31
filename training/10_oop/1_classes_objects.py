"""
Introduction to classes and objects in Python.
"""

class Person:
    """A simple class to represent a person."""
    
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        """Initialize a new person."""
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
        self._private_var = "I'm private"  # Convention for private variable
    
    def introduce(self):
        """Instance method to introduce the person."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    def have_birthday(self):
        """Increment the person's age."""
        self.age += 1
        return f"Happy birthday! {self.name} is now {self.age}."
    
    @property
    def description(self):
        """Property decorator example."""
        return f"{self.name} ({self.age} years old)"
    
    @classmethod
    def create_from_birth_year(cls, name, birth_year):
        """Class method to create a Person from birth year."""
        from datetime import datetime
        age = datetime.now().year - birth_year
        return cls(name, age)
    
    @staticmethod
    def is_adult(age):
        """Static method to check if age is adult."""
        return age >= 18

# Example usage
if __name__ == "__main__":
    # Creating instances
    person1 = Person("Alice", 25)
    person2 = Person("Bob", 30)
    
    # Using instance methods
    print(person1.introduce())
    print(person2.introduce())
    
    # Accessing class variable
    print(f"\nSpecies: {Person.species}")
    print(f"Person 1's species: {person1.species}")
    
    # Using property
    print(f"\nDescription: {person1.description}")
    
    # Using instance method
    print(person1.have_birthday())
    
    # Using class method
    person3 = Person.create_from_birth_year("Charlie", 1995)
    print(person3.introduce())
    
    # Using static method
    print(f"\nIs 20 adult? {Person.is_adult(20)}")
    print(f"Is 15 adult? {Person.is_adult(15)}")
    
    # Demonstrating attribute access
    print(f"\nDirect attribute access: {person1.name}")
    
    # Demonstrating instance checks
    print(f"\nIs person1 a Person? {isinstance(person1, Person)}")
    print(f"Is Person a type? {isinstance(Person, type)}") 