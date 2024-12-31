"""
Demonstration of inheritance and polymorphism in Python.
"""

class Animal:
    """Base class for animals."""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        """Base method for making sound."""
        return "Some generic sound"
    
    def describe(self):
        """Describe the animal."""
        return f"{self.name} is a {self.species}"
    
    @classmethod
    def create_anonymous(cls, species):
        """Create an animal without a name."""
        return cls("Unknown", species)

class Dog(Animal):
    """Dog class inheriting from Animal."""
    
    def __init__(self, name, breed):
        """Initialize a dog with breed instead of species."""
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self):
        """Override the make_sound method."""
        return "Woof!"
    
    def fetch(self):
        """Dog-specific method."""
        return f"{self.name} is fetching the ball"
    
    def describe(self):
        """Override describe to include breed."""
        return f"{super().describe()} of breed {self.breed}"

class Cat(Animal):
    """Cat class inheriting from Animal."""
    
    def __init__(self, name, indoor=True):
        """Initialize a cat with indoor/outdoor status."""
        super().__init__(name, species="Cat")
        self.indoor = indoor
    
    def make_sound(self):
        """Override the make_sound method."""
        return "Meow!"
    
    def scratch(self):
        """Cat-specific method."""
        return f"{self.name} is scratching"
    
    @property
    def lifestyle(self):
        """Property to describe indoor/outdoor status."""
        return "indoor" if self.indoor else "outdoor"

class GermanShepherd(Dog):
    """Specific dog breed class demonstrating multi-level inheritance."""
    
    def __init__(self, name):
        """Initialize a German Shepherd."""
        super().__init__(name, breed="German Shepherd")
    
    def guard(self):
        """Breed-specific method."""
        return f"{self.name} is guarding"

def demonstrate_polymorphism(animal):
    """Demonstrate polymorphic behavior."""
    print(f"\nDemonstrating polymorphism with {animal.name}:")
    print(f"Description: {animal.describe()}")
    print(f"Sound: {animal.make_sound()}")

# Example usage
if __name__ == "__main__":
    # Create instances
    generic_animal = Animal("Generic", "Unknown")
    dog = Dog("Buddy", "Golden Retriever")
    cat = Cat("Whiskers", indoor=True)
    shepherd = GermanShepherd("Rex")
    
    # Demonstrate inheritance
    print("Basic animal:", generic_animal.describe())
    print("Dog:", dog.describe())
    print("Cat:", cat.describe())
    print("German Shepherd:", shepherd.describe())
    
    # Demonstrate method overriding
    print("\nMaking sounds:")
    print(f"{generic_animal.name}:", generic_animal.make_sound())
    print(f"{dog.name}:", dog.make_sound())
    print(f"{cat.name}:", cat.make_sound())
    
    # Demonstrate specific methods
    print("\nSpecific behaviors:")
    print(dog.fetch())
    print(cat.scratch())
    print(shepherd.guard())
    
    # Demonstrate properties
    print(f"\n{cat.name} is an {cat.lifestyle} cat")
    
    # Demonstrate polymorphism
    animals = [generic_animal, dog, cat, shepherd]
    for animal in animals:
        demonstrate_polymorphism(animal)
    
    # Demonstrate isinstance checks
    print("\nType checking:")
    print(f"Is {dog.name} a Dog?", isinstance(dog, Dog))
    print(f"Is {dog.name} an Animal?", isinstance(dog, Animal))
    print(f"Is {shepherd.name} a Dog?", isinstance(shepherd, Dog))
    print(f"Is {cat.name} a Dog?", isinstance(cat, Dog)) 