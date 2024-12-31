"""
Dictionaries in Python: key-value pairs and operations.
"""

# Creating dictionaries
person = {
    "name": "John",
    "age": 30,
    "occupation": "developer",
    "skills": ["Python", "JavaScript"]
}

# Basic dictionary operations
print("Dictionary Operations:")
print(f"Dictionary: {person}")
print(f"Name: {person['name']}")
print(f"Age: {person.get('age')}")

# Adding and modifying entries
person['location'] = 'New York'
person['age'] = 31
print(f"\nAfter modifications: {person}")

# Dictionary methods
print("\nDictionary Methods:")
print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Items: {list(person.items())}")

# Nested dictionaries
company = {
    "name": "Tech Corp",
    "employees": {
        "john": {"role": "developer", "years": 5},
        "alice": {"role": "manager", "years": 7}
    }
}
print(f"\nNested dictionary: {company}")
print(f"John's role: {company['employees']['john']['role']}") 