"""
Demonstration of additional design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from copy import deepcopy

# Decorator Pattern
class Coffee(ABC):
    """Abstract coffee class."""
    
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass

class SimpleCoffee(Coffee):
    """Basic coffee implementation."""
    
    def cost(self) -> float:
        return 2.0
    
    def description(self) -> str:
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    """Base decorator class."""
    
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

class MilkDecorator(CoffeeDecorator):
    """Concrete decorator for adding milk."""
    
    def cost(self) -> float:
        return self._coffee.cost() + 0.5
    
    def description(self) -> str:
        return f"{self._coffee.description()} with milk"

class SugarDecorator(CoffeeDecorator):
    """Concrete decorator for adding sugar."""
    
    def cost(self) -> float:
        return self._coffee.cost() + 0.2
    
    def description(self) -> str:
        return f"{self._coffee.description()} with sugar"

# Prototype Pattern
class Prototype:
    """Base prototype class."""
    
    def clone(self):
        """Create a deep copy of the object."""
        return deepcopy(self)

class Document(Prototype):
    """Document class with prototype capability."""
    
    def __init__(self, content: str, formatting: Dict[str, Any]):
        self.content = content
        self.formatting = formatting
    
    def __str__(self):
        return f"Content: {self.content}\nFormatting: {self.formatting}"

# Strategy Pattern
class SortStrategy(ABC):
    """Abstract sorting strategy."""
    
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

class BubbleSort(SortStrategy):
    """Bubble sort implementation."""
    
    def sort(self, data: List[int]) -> List[int]:
        result = data.copy()
        n = len(result)
        for i in range(n):
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result

class QuickSort(SortStrategy):
    """Quick sort implementation."""
    
    def sort(self, data: List[int]) -> List[int]:
        result = data.copy()
        if len(result) <= 1:
            return result
        pivot = result[len(result) // 2]
        left = [x for x in result if x < pivot]
        middle = [x for x in result if x == pivot]
        right = [x for x in result if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    """Context class for sorting strategy."""
    
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy):
        """Change the sorting strategy."""
        self._strategy = strategy
    
    def sort(self, data: List[int]) -> List[int]:
        """Sort data using the current strategy."""
        return self._strategy.sort(data)

# State Pattern
class VendingMachineState(ABC):
    """Abstract state for vending machine."""
    
    @abstractmethod
    def insert_coin(self, machine: 'VendingMachine') -> None:
        pass
    
    @abstractmethod
    def eject_coin(self, machine: 'VendingMachine') -> None:
        pass
    
    @abstractmethod
    def dispense(self, machine: 'VendingMachine') -> None:
        pass

class NoCoinState(VendingMachineState):
    """State when no coin is inserted."""
    
    def insert_coin(self, machine: 'VendingMachine') -> None:
        print("Coin inserted")
        machine.state = HasCoinState()
    
    def eject_coin(self, machine: 'VendingMachine') -> None:
        print("No coin to eject")
    
    def dispense(self, machine: 'VendingMachine') -> None:
        print("Please insert a coin first")

class HasCoinState(VendingMachineState):
    """State when coin is inserted."""
    
    def insert_coin(self, machine: 'VendingMachine') -> None:
        print("Coin already inserted")
    
    def eject_coin(self, machine: 'VendingMachine') -> None:
        print("Coin ejected")
        machine.state = NoCoinState()
    
    def dispense(self, machine: 'VendingMachine') -> None:
        print("Item dispensed")
        machine.state = NoCoinState()

class VendingMachine:
    """Vending machine using state pattern."""
    
    def __init__(self):
        self.state = NoCoinState()
    
    def insert_coin(self):
        self.state.insert_coin(self)
    
    def eject_coin(self):
        self.state.eject_coin(self)
    
    def dispense(self):
        self.state.dispense(self)

# Example usage
if __name__ == "__main__":
    # Demonstrate Decorator Pattern
    print("Decorator Pattern:")
    coffee = SimpleCoffee()
    coffee_with_milk = MilkDecorator(coffee)
    coffee_with_milk_and_sugar = SugarDecorator(coffee_with_milk)
    
    print(f"{coffee_with_milk_and_sugar.description()}")
    print(f"Cost: ${coffee_with_milk_and_sugar.cost():.2f}")
    
    # Demonstrate Prototype Pattern
    print("\nPrototype Pattern:")
    doc = Document("Hello", {"font": "Arial", "size": 12})
    doc_copy = doc.clone()
    doc_copy.formatting["size"] = 14
    
    print("Original:", doc)
    print("Clone:", doc_copy)
    
    # Demonstrate Strategy Pattern
    print("\nStrategy Pattern:")
    data = [64, 34, 25, 12, 22, 11, 90]
    sorter = Sorter(BubbleSort())
    print("Bubble sort:", sorter.sort(data))
    
    sorter.set_strategy(QuickSort())
    print("Quick sort:", sorter.sort(data))
    
    # Demonstrate State Pattern
    print("\nState Pattern:")
    machine = VendingMachine()
    machine.dispense()  # Should fail
    machine.insert_coin()
    machine.insert_coin()  # Should fail
    machine.dispense()
    machine.dispense()  # Should fail 