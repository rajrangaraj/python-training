"""
Real-world examples using conditional statements.
"""

def calculate_discount(cart_total, is_member, coupon_code=None):
    """
    Calculate discount based on various conditions.
    """
    discount = 0
    
    # Member discount
    if is_member:
        if cart_total >= 200:
            discount = 0.20  # 20% off for members spending $200+
        elif cart_total >= 100:
            discount = 0.15  # 15% off for members spending $100+
        else:
            discount = 0.10  # 10% off for members
    
    # Coupon discount
    if coupon_code:
        if coupon_code == "SAVE30" and cart_total >= 150:
            discount = max(discount, 0.30)
        elif coupon_code == "SAVE20":
            discount = max(discount, 0.20)
    
    final_price = cart_total * (1 - discount)
    savings = cart_total - final_price
    
    return {
        "original_price": cart_total,
        "discount_percentage": discount * 100,
        "savings": savings,
        "final_price": final_price
    }

def determine_shipping(country, order_total, is_express=False):
    """
    Calculate shipping cost and delivery time based on various factors.
    """
    if country.upper() == "US":
        if order_total >= 100:
            shipping_cost = 0  # Free shipping for orders over $100
        elif is_express:
            shipping_cost = 25
        else:
            shipping_cost = 10
        delivery_days = 2 if is_express else 5
    
    elif country.upper() in ["CA", "MX"]:  # Canada or Mexico
        if order_total >= 150:
            shipping_cost = 15
        elif is_express:
            shipping_cost = 45
        else:
            shipping_cost = 30
        delivery_days = 4 if is_express else 8
    
    else:  # International
        if order_total >= 200:
            shipping_cost = 30
        elif is_express:
            shipping_cost = 80
        else:
            shipping_cost = 50
        delivery_days = 7 if is_express else 14
    
    return {
        "shipping_cost": shipping_cost,
        "delivery_days": delivery_days,
        "free_shipping": shipping_cost == 0
    }

# Example usage
if __name__ == "__main__":
    # Test discount calculator
    print("Discount Examples:")
    print("\nMember with $250 purchase:")
    result = calculate_discount(250, True, "SAVE30")
    for key, value in result.items():
        print(f"{key}: ${value:.2f}")
    
    print("\nNon-member with $150 purchase and coupon:")
    result = calculate_discount(150, False, "SAVE20")
    for key, value in result.items():
        print(f"{key}: ${value:.2f}")
    
    # Test shipping calculator
    print("\nShipping Examples:")
    print("\nUS Order:")
    result = determine_shipping("US", 120, False)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print("\nInternational Express Order:")
    result = determine_shipping("UK", 175, True)
    for key, value in result.items():
        print(f"{key}: {value}") 