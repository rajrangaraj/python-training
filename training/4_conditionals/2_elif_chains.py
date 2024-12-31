"""
Examples of if-elif-else chains in Python.
"""

def grade_calculator(score):
    """Calculate letter grade based on numerical score."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def time_of_day_greeting(hour):
    """Return appropriate greeting based on hour of day."""
    if hour < 0 or hour > 23:
        return "Invalid hour"
    elif hour < 6:
        return "Good night"
    elif hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    elif hour < 22:
        return "Good evening"
    else:
        return "Good night"

def weather_advice(temperature, is_raining):
    """Provide weather-appropriate advice."""
    if temperature > 35:
        advice = "It's very hot! Stay hydrated"
    elif temperature > 25:
        advice = "It's warm"
    elif temperature > 15:
        advice = "It's mild"
    elif temperature > 5:
        advice = "It's cool"
    else:
        advice = "It's cold! Wear warm clothes"
    
    if is_raining:
        advice += " and bring an umbrella!"
    
    return advice

# Example usage
if __name__ == "__main__":
    # Test grade calculator
    print(f"Score 85: {grade_calculator(85)}")
    
    # Test time greeting
    print(f"Greeting at 14:00: {time_of_day_greeting(14)}")
    
    # Test weather advice
    print(f"Weather advice: {weather_advice(28, True)}") 