"""
Demonstration of final set of advanced design patterns in Python.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Template Method Pattern
class DataMiner(ABC):
    """Abstract class defining template method for data mining."""
    
    def mine(self, path: str) -> Dict[str, Any]:
        """Template method defining the data mining algorithm."""
        raw_data = self._extract_data(path)
        parsed_data = self._parse_data(raw_data)
        analyzed_data = self._analyze_data(parsed_data)
        report = self._generate_report(analyzed_data)
        return report
    
    @abstractmethod
    def _extract_data(self, path: str) -> str:
        """Extract data from source."""
        pass
    
    @abstractmethod
    def _parse_data(self, data: str) -> List[Any]:
        """Parse the extracted data."""
        pass
    
    def _analyze_data(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze the parsed data (common implementation)."""
        return {
            "count": len(data),
            "first": data[0] if data else None,
            "last": data[-1] if data else None
        }
    
    def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report (common implementation)."""
        data["timestamp"] = datetime.now().isoformat()
        return data

class PDFMiner(DataMiner):
    """Concrete implementation for mining PDF files."""
    
    def _extract_data(self, path: str) -> str:
        print(f"Extracting data from PDF: {path}")
        return "PDF raw content"
    
    def _parse_data(self, data: str) -> List[Any]:
        print("Parsing PDF data")
        return ["PDF", "parsed", "content"]

class CSVMiner(DataMiner):
    """Concrete implementation for mining CSV files."""
    
    def _extract_data(self, path: str) -> str:
        print(f"Extracting data from CSV: {path}")
        return "CSV raw content"
    
    def _parse_data(self, data: str) -> List[Any]:
        print("Parsing CSV data")
        return ["CSV", "parsed", "content"]

# Memento Pattern
class EditorMemento:
    """Memento class storing editor state."""
    
    def __init__(self, content: str):
        self._content = content
    
    def get_content(self) -> str:
        return self._content

class Editor:
    """Originator class for text editing."""
    
    def __init__(self):
        self._content = ""
    
    def write(self, text: str) -> None:
        """Add text to the editor."""
        self._content += text
    
    def get_content(self) -> str:
        """Get current content."""
        return self._content
    
    def save(self) -> EditorMemento:
        """Save current state."""
        return EditorMemento(self._content)
    
    def restore(self, memento: EditorMemento) -> None:
        """Restore to previous state."""
        self._content = memento.get_content()

class History:
    """Caretaker class managing editor history."""
    
    def __init__(self):
        self._mementos: List[EditorMemento] = []
    
    def push(self, memento: EditorMemento) -> None:
        """Add a memento to history."""
        self._mementos.append(memento)
    
    def pop(self) -> Optional[EditorMemento]:
        """Get the last memento."""
        if self._mementos:
            return self._mementos.pop()
        return None

# Visitor Pattern
class ReportElement(ABC):
    """Abstract element that can be visited."""
    
    @abstractmethod
    def accept(self, visitor: 'ReportVisitor') -> None:
        pass

class IncomeData(ReportElement):
    """Concrete element representing income data."""
    
    def __init__(self, amount: float):
        self.amount = amount
    
    def accept(self, visitor: 'ReportVisitor') -> None:
        visitor.visit_income(self)

class ExpenseData(ReportElement):
    """Concrete element representing expense data."""
    
    def __init__(self, amount: float):
        self.amount = amount
    
    def accept(self, visitor: 'ReportVisitor') -> None:
        visitor.visit_expense(self)

class ReportVisitor(ABC):
    """Abstract visitor for generating reports."""
    
    @abstractmethod
    def visit_income(self, income: IncomeData) -> None:
        pass
    
    @abstractmethod
    def visit_expense(self, expense: ExpenseData) -> None:
        pass

class JSONReportVisitor(ReportVisitor):
    """Concrete visitor generating JSON reports."""
    
    def __init__(self):
        self.data = {"income": [], "expenses": []}
    
    def visit_income(self, income: IncomeData) -> None:
        self.data["income"].append(income.amount)
    
    def visit_expense(self, expense: ExpenseData) -> None:
        self.data["expenses"].append(expense.amount)
    
    def get_report(self) -> str:
        return json.dumps(self.data, indent=2)

class TextReportVisitor(ReportVisitor):
    """Concrete visitor generating text reports."""
    
    def __init__(self):
        self.report_lines = []
    
    def visit_income(self, income: IncomeData) -> None:
        self.report_lines.append(f"Income: ${income.amount:.2f}")
    
    def visit_expense(self, expense: ExpenseData) -> None:
        self.report_lines.append(f"Expense: ${expense.amount:.2f}")
    
    def get_report(self) -> str:
        return "\n".join(self.report_lines)

# Example usage
if __name__ == "__main__":
    # Demonstrate Template Method Pattern
    print("Template Method Pattern:")
    miners = [PDFMiner(), CSVMiner()]
    for miner in miners:
        result = miner.mine(f"data.{miner.__class__.__name__.lower()}")
        print(f"Mining result: {result}\n")
    
    # Demonstrate Memento Pattern
    print("Memento Pattern:")
    editor = Editor()
    history = History()
    
    # Make some edits and save states
    editor.write("First line\n")
    history.push(editor.save())
    
    editor.write("Second line\n")
    history.push(editor.save())
    
    editor.write("Third line\n")
    print("Current content:")
    print(editor.get_content())
    
    # Undo changes
    print("\nUndoing last change:")
    if last_save := history.pop():
        editor.restore(last_save)
        print(editor.get_content())
    
    # Demonstrate Visitor Pattern
    print("\nVisitor Pattern:")
    financial_data = [
        IncomeData(1000.0),
        ExpenseData(500.0),
        IncomeData(750.0),
        ExpenseData(250.0)
    ]
    
    # Generate JSON report
    json_visitor = JSONReportVisitor()
    for data in financial_data:
        data.accept(json_visitor)
    print("JSON Report:")
    print(json_visitor.get_report())
    
    # Generate Text report
    text_visitor = TextReportVisitor()
    for data in financial_data:
        data.accept(text_visitor)
    print("\nText Report:")
    print(text_visitor.get_report()) 