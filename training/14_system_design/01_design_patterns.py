"""
Demonstration of common design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass
import threading
from datetime import datetime
import json

# Singleton Pattern
class Singleton:
    """
    Singleton pattern ensures a class has only one instance.
    Useful for managing global state or resources.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.config = {}
    
    def set_config(self, key: str, value: Any):
        self.config[key] = value
    
    def get_config(self, key: str) -> Any:
        return self.config.get(key)

# Factory Pattern
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

class AnimalFactory:
    """
    Factory pattern creates objects without exposing creation logic.
    """
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Observer Pattern
class Observer(ABC):
    @abstractmethod
    def update(self, message: str):
        pass

class Subject:
    """
    Observer pattern allows objects to subscribe to changes.
    """
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify(f"State changed to: {value}")

# Strategy Pattern
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount: float) -> bool:
        print(f"Paying ${amount} with Credit Card")
        return True

class PayPalPayment(PaymentStrategy):
    def pay(self, amount: float) -> bool:
        print(f"Paying ${amount} with PayPal")
        return True

class ShoppingCart:
    """
    Strategy pattern enables selecting algorithms at runtime.
    """
    def __init__(self):
        self.items: List[Dict] = []
        self.payment_strategy: PaymentStrategy = None
    
    def add_item(self, item: Dict):
        self.items.append(item)
    
    def set_payment_strategy(self, strategy: PaymentStrategy):
        self.payment_strategy = strategy
    
    def checkout(self) -> bool:
        if not self.payment_strategy:
            raise ValueError("Payment strategy not set")
        
        total = sum(item['price'] for item in self.items)
        return self.payment_strategy.pay(total)

# Decorator Pattern
@dataclass
class Coffee:
    cost: float = 3.0
    description: str = "Simple coffee"

class CoffeeDecorator(Coffee):
    """
    Decorator pattern adds behavior to objects dynamically.
    """
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    
    @property
    def cost(self) -> float:
        return self._coffee.cost
    
    @property
    def description(self) -> str:
        return self._coffee.description

class MilkDecorator(CoffeeDecorator):
    @property
    def cost(self) -> float:
        return self._coffee.cost + 0.5
    
    @property
    def description(self) -> str:
        return f"{self._coffee.description}, milk"

class SugarDecorator(CoffeeDecorator):
    @property
    def cost(self) -> float:
        return self._coffee.cost + 0.2
    
    @property
    def description(self) -> str:
        return f"{self._coffee.description}, sugar"

def demonstrate_patterns():
    """Demonstrate the usage of various design patterns."""
    
    print("\n1. Singleton Pattern:")
    config1 = Singleton()
    config1.set_config("server", "localhost")
    
    config2 = Singleton()
    print(f"Same instance? {config1 is config2}")
    print(f"Config value: {config2.get_config('server')}")
    
    print("\n2. Factory Pattern:")
    factory = AnimalFactory()
    dog = factory.create_animal("dog")
    cat = factory.create_animal("cat")
    print(f"Dog says: {dog.speak()}")
    print(f"Cat says: {cat.speak()}")
    
    print("\n3. Observer Pattern:")
    class LogObserver(Observer):
        def update(self, message: str):
            print(f"Log: {message}")
    
    subject = Subject()
    observer = LogObserver()
    subject.attach(observer)
    subject.state = "active"
    
    print("\n4. Strategy Pattern:")
    cart = ShoppingCart()
    cart.add_item({"name": "Book", "price": 20.0})
    cart.set_payment_strategy(PayPalPayment())
    cart.checkout()
    
    print("\n5. Decorator Pattern:")
    coffee = Coffee()
    coffee_with_milk = MilkDecorator(coffee)
    coffee_with_milk_sugar = SugarDecorator(coffee_with_milk)
    print(f"Order: {coffee_with_milk_sugar.description}")
    print(f"Cost: ${coffee_with_milk_sugar.cost:.2f}")

if __name__ == "__main__":
    demonstrate_patterns() 