"""
Real-world examples using arithmetic operations.
"""

def calculate_circle_properties(radius):
    """Calculate various properties of a circle."""
    import math
    
    area = math.pi * radius ** 2
    circumference = 2 * math.pi * radius
    diameter = 2 * radius
    
    print(f"Circle Properties (radius = {radius}):")
    print(f"Area: {area:.2f}")
    print(f"Circumference: {circumference:.2f}")
    print(f"Diameter: {diameter}")

def calculate_investment_growth(principal, rate, years):
    """Calculate investment growth with compound interest."""
    amount = principal * (1 + rate) ** years
    interest_earned = amount - principal
    
    print(f"\nInvestment Growth:")
    print(f"Principal: ${principal:,.2f}")
    print(f"Rate: {rate:.1%}")
    print(f"Years: {years}")
    print(f"Final Amount: ${amount:,.2f}")
    print(f"Interest Earned: ${interest_earned:,.2f}")

def calculate_monthly_payment(loan_amount, annual_rate, years):
    """Calculate monthly mortgage payment."""
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
             ((1 + monthly_rate)**num_payments - 1)
    
    print(f"\nMortgage Calculator:")
    print(f"Loan Amount: ${loan_amount:,.2f}")
    print(f"Annual Rate: {annual_rate:.1%}")
    print(f"Years: {years}")
    print(f"Monthly Payment: ${payment:,.2f}")

# Run examples
if __name__ == "__main__":
    calculate_circle_properties(5)
    calculate_investment_growth(10000, 0.07, 10)
    calculate_monthly_payment(200000, 0.035, 30) 