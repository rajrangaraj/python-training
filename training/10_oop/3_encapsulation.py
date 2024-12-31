"""
Demonstration of encapsulation, properties, and access control in Python.
"""

class BankAccount:
    """A class demonstrating encapsulation with a bank account."""
    
    def __init__(self, account_holder, initial_balance=0):
        self._account_holder = account_holder  # Protected attribute
        self.__balance = initial_balance       # Private attribute
        self.__transaction_history = []        # Private attribute
    
    @property
    def balance(self):
        """Get the current balance."""
        return self.__balance
    
    @property
    def account_holder(self):
        """Get the account holder's name."""
        return self._account_holder
    
    @account_holder.setter
    def account_holder(self, name):
        """Set the account holder's name."""
        if not name.strip():
            raise ValueError("Account holder name cannot be empty")
        self._account_holder = name.strip()
    
    def deposit(self, amount):
        """Deposit money into the account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.__balance += amount
        self.__record_transaction("deposit", amount)
        return f"Deposited ${amount}. New balance: ${self.__balance}"
    
    def withdraw(self, amount):
        """Withdraw money from the account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        
        self.__balance -= amount
        self.__record_transaction("withdrawal", amount)
        return f"Withdrew ${amount}. New balance: ${self.__balance}"
    
    def get_transaction_history(self):
        """Get a copy of the transaction history."""
        return self.__transaction_history.copy()
    
    def __record_transaction(self, transaction_type, amount):
        """Private method to record transactions."""
        from datetime import datetime
        transaction = {
            'type': transaction_type,
            'amount': amount,
            'timestamp': datetime.now(),
            'balance': self.__balance
        }
        self.__transaction_history.append(transaction)

class SavingsAccount(BankAccount):
    """A savings account with interest rate."""
    
    def __init__(self, account_holder, initial_balance=0, interest_rate=0.01):
        super().__init__(account_holder, initial_balance)
        self.__interest_rate = interest_rate
    
    @property
    def interest_rate(self):
        """Get the current interest rate."""
        return self.__interest_rate
    
    @interest_rate.setter
    def interest_rate(self, rate):
        """Set the interest rate."""
        if rate < 0 or rate > 0.1:  # Max 10% interest
            raise ValueError("Interest rate must be between 0 and 0.1")
        self.__interest_rate = rate
    
    def apply_interest(self):
        """Apply interest to the account."""
        interest = self.balance * self.__interest_rate
        self.deposit(interest)
        return f"Applied interest: ${interest:.2f}"

# Example usage
if __name__ == "__main__":
    # Create a basic bank account
    account = BankAccount("John Doe", 1000)
    
    # Demonstrate property access
    print(f"Account holder: {account.account_holder}")
    print(f"Initial balance: ${account.balance}")
    
    # Demonstrate encapsulation
    try:
        # This will raise an AttributeError
        print(account.__balance)
    except AttributeError as e:
        print(f"Cannot access private attribute: {e}")
    
    # Perform some transactions
    print("\nPerforming transactions:")
    print(account.deposit(500))
    print(account.withdraw(200))
    
    # Show transaction history
    print("\nTransaction history:")
    for transaction in account.get_transaction_history():
        print(f"{transaction['type'].capitalize()}: ${transaction['amount']}")
    
    # Create a savings account
    savings = SavingsAccount("Jane Doe", 2000, 0.05)
    print(f"\nSavings account created for {savings.account_holder}")
    print(f"Initial balance: ${savings.balance}")
    print(f"Interest rate: {savings.interest_rate * 100}%")
    
    # Apply interest
    print(savings.apply_interest())
    print(f"New balance after interest: ${savings.balance}")
    
    # Demonstrate property setter validation
    try:
        savings.interest_rate = 0.2  # This will raise ValueError
    except ValueError as e:
        print(f"\nError setting interest rate: {e}") 