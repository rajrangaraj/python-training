"""
Real-world examples using different Python data types.
"""

# Student grade tracking system
def track_student_grades():
    student = {
        "name": "Alice Smith",
        "grades": [85, 92, 78, 90, 88],
        "subjects": ["Math", "English", "Science", "History", "Art"],
        "active": True
    }
    
    # Calculate average grade
    average = sum(student["grades"]) / len(student["grades"])
    
    # Create grade summary
    grade_summary = {
        subject: grade
        for subject, grade in zip(student["subjects"], student["grades"])
    }
    
    print("Student Grade Summary:")
    print(f"Name: {student['name']}")
    print(f"Average Grade: {average:.2f}")
    print("Grades by Subject:")
    for subject, grade in grade_summary.items():
        print(f"  {subject}: {grade}")

# Shopping cart system
def process_shopping_cart():
    cart = [
        {"item": "Laptop", "price": 999.99, "quantity": 1},
        {"item": "Mouse", "price": 25.50, "quantity": 2},
        {"item": "Keyboard", "price": 50.00, "quantity": 1}
    ]
    
    # Calculate totals
    subtotal = sum(item["price"] * item["quantity"] for item in cart)
    tax_rate = 0.08
    tax = subtotal * tax_rate
    total = subtotal + tax
    
    print("\nShopping Cart Summary:")
    for item in cart:
        print(f"{item['item']}: ${item['price']} x {item['quantity']}")
    print(f"Subtotal: ${subtotal:.2f}")
    print(f"Tax (8%): ${tax:.2f}")
    print(f"Total: ${total:.2f}")

# Run examples
if __name__ == "__main__":
    track_student_grades()
    process_shopping_cart() 