"""
Working with JSON files in Python.
"""

import json
from datetime import datetime
import os

def write_basic_json():
    """Write basic data to a JSON file."""
    # Sample data
    data = {
        'name': 'John Doe',
        'age': 30,
        'city': 'New York',
        'interests': ['programming', 'reading', 'hiking'],
        'is_student': False
    }
    
    # Write to JSON file
    with open('person.json', 'w') as file:
        json.dump(data, file, indent=4)
    
    print("Basic JSON file written")

def read_basic_json():
    """Read data from a JSON file."""
    print("\nReading basic JSON:")
    with open('person.json', 'r') as file:
        data = json.load(file)
        print(f"Loaded data: {data}")
        print(f"Name: {data['name']}")
        print(f"Interests: {', '.join(data['interests'])}")

def write_complex_json():
    """Write more complex data to JSON."""
    # Complex data structure
    data = {
        'company': 'Tech Corp',
        'employees': [
            {
                'id': 1,
                'name': 'Alice Brown',
                'department': 'Engineering',
                'projects': ['Project A', 'Project B'],
                'details': {
                    'hire_date': '2022-01-15',
                    'salary': 75000,
                    'is_manager': True
                }
            },
            {
                'id': 2,
                'name': 'Bob Smith',
                'department': 'Marketing',
                'projects': ['Project C'],
                'details': {
                    'hire_date': '2021-06-01',
                    'salary': 65000,
                    'is_manager': False
                }
            }
        ],
        'metadata': {
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    # Write to JSON file
    with open('company.json', 'w') as file:
        json.dump(data, file, indent=4)
    
    print("\nComplex JSON file written")

def read_complex_json():
    """Read and process complex JSON data."""
    print("\nReading complex JSON:")
    with open('company.json', 'r') as file:
        data = json.load(file)
        
        print(f"Company: {data['company']}")
        print("\nEmployees:")
        for employee in data['employees']:
            print(f"\nEmployee ID: {employee['id']}")
            print(f"Name: {employee['name']}")
            print(f"Department: {employee['department']}")
            print(f"Projects: {', '.join(employee['projects'])}")
            print(f"Hire Date: {employee['details']['hire_date']}")
            print(f"Is Manager: {employee['details']['is_manager']}")
        
        print(f"\nLast Updated: {data['metadata']['last_updated']}")

def update_json_file():
    """Demonstrate updating a JSON file."""
    # Read existing data
    with open('person.json', 'r') as file:
        data = json.load(file)
    
    # Modify data
    data['age'] += 1  # Increment age
    data['interests'].append('cooking')  # Add new interest
    data['last_updated'] = datetime.now().isoformat()
    
    # Write back to file
    with open('person.json', 'w') as file:
        json.dump(data, file, indent=4)
    
    print("\nJSON file updated")

# Example usage
if __name__ == "__main__":
    write_basic_json()
    read_basic_json()
    
    write_complex_json()
    read_complex_json()
    
    update_json_file()
    print("\nAfter update:")
    read_basic_json() 