"""
Real-world examples using loops.
"""

def print_multiplication_table(n):
    """Print multiplication table for numbers 1 to n."""
    print(f"\nMultiplication Table (1 to {n}):")
    
    # Print header
    print("   ", end="")
    for i in range(1, n + 1):
        print(f"{i:4}", end="")
    print("\n" + "-" * (n * 4 + 4))
    
    # Print table
    for i in range(1, n + 1):
        print(f"{i:2} |", end="")
        for j in range(1, n + 1):
            print(f"{i*j:4}", end="")
        print()

def process_sales_data():
    """Process daily sales data."""
    sales_data = [
        ("Monday", 150),
        ("Tuesday", 200),
        ("Wednesday", 125),
        ("Thursday", 175),
        ("Friday", 225)
    ]
    
    total = 0
    highest = ("", 0)
    lowest = ("", float('inf'))
    
    print("\nSales Report:")
    print("-" * 20)
    
    for day, amount in sales_data:
        print(f"{day}: ${amount}")
        total += amount
        
        if amount > highest[1]:
            highest = (day, amount)
        if amount < lowest[1]:
            lowest = (day, amount)
    
    average = total / len(sales_data)
    print(f"\nTotal Sales: ${total}")
    print(f"Average Daily Sales: ${average:.2f}")
    print(f"Best Day: {highest[0]} (${highest[1]})")
    print(f"Worst Day: {lowest[0]} (${lowest[1]})")

def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    print(f"\nFibonacci Sequence (first {n} terms):")
    a, b = 0, 1
    count = 0
    
    while count < n:
        print(a, end=" ")
        a, b = b, a + b
        count += 1
    print()

# Run examples
if __name__ == "__main__":
    print_multiplication_table(5)
    process_sales_data()
    fibonacci_sequence(10) 