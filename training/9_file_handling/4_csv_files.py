"""
Working with CSV (Comma-Separated Values) files in Python.
"""

import csv
from datetime import datetime

def write_basic_csv():
    """Write data to a basic CSV file."""
    # Sample data
    headers = ['Name', 'Age', 'City']
    data = [
        ['John Doe', 30, 'New York'],
        ['Jane Smith', 25, 'Los Angeles'],
        ['Bob Johnson', 35, 'Chicago']
    ]
    
    # Write to CSV
    with open('people.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    
    print("Basic CSV file written")

def read_basic_csv():
    """Read data from a basic CSV file."""
    print("\nReading basic CSV:")
    with open('people.csv', 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        print(f"Headers: {headers}")
        
        for row in reader:
            print(f"Row: {row}")

def write_dict_csv():
    """Write data using dictionary format."""
    # Sample data
    data = [
        {'name': 'Alice Brown', 'age': 28, 'city': 'Boston'},
        {'name': 'Charlie Davis', 'age': 32, 'city': 'Seattle'},
        {'name': 'Eva Wilson', 'age': 24, 'city': 'Miami'}
    ]
    
    # Write to CSV
    with open('people_dict.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print("\nDictionary-based CSV file written")

def read_dict_csv():
    """Read data using dictionary format."""
    print("\nReading dictionary-based CSV:")
    with open('people_dict.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(f"Person: {dict(row)}")

def write_complex_csv():
    """Write CSV with more complex data and custom formatting."""
    # Sample data with different types
    data = [
        {
            'id': 1,
            'name': 'Product A',
            'price': 19.99,
            'date_added': datetime.now(),
            'in_stock': True
        },
        {
            'id': 2,
            'name': 'Product B',
            'price': 29.99,
            'date_added': datetime.now(),
            'in_stock': False
        }
    ]
    
    # Write to CSV with custom formatting
    with open('products.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        
        for row in data:
            # Format datetime and boolean values
            row['date_added'] = row['date_added'].strftime('%Y-%m-%d %H:%M:%S')
            row['in_stock'] = 'Yes' if row['in_stock'] else 'No'
            writer.writerow(row)
    
    print("\nComplex CSV file written")

def read_complex_csv():
    """Read and parse complex CSV data."""
    print("\nReading complex CSV:")
    with open('products.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert string values back to appropriate types
            row['id'] = int(row['id'])
            row['price'] = float(row['price'])
            row['date_added'] = datetime.strptime(
                row['date_added'], 
                '%Y-%m-%d %H:%M:%S'
            )
            row['in_stock'] = row['in_stock'] == 'Yes'
            print(f"Product: {dict(row)}")

# Example usage
if __name__ == "__main__":
    write_basic_csv()
    read_basic_csv()
    
    write_dict_csv()
    read_dict_csv()
    
    write_complex_csv()
    read_complex_csv() 