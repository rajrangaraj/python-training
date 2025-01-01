"""
Demonstration of test coverage analysis and property-based testing using hypothesis.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import pytest
from hypothesis import given, strategies as st
import coverage

@dataclass
class BankAccount:
    """Bank account class for demonstrating comprehensive testing."""
    
    account_id: str
    balance: float
    overdraft_limit: float = 0.0
    
    def deposit(self, amount: float) -> bool:
        """Deposit money into the account."""
        if amount <= 0:
            return False
        self.balance += amount
        return True
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw money from the account if sufficient funds available."""
        if amount <= 0:
            return False
        
        if self.balance - amount >= -self.overdraft_limit:
            self.balance -= amount
            return True
        return False
    
    def transfer(self, other: 'BankAccount', amount: float) -> bool:
        """Transfer money to another account."""
        if self.withdraw(amount):
            if other.deposit(amount):
                return True
            # Rollback if deposit fails
            self.deposit(amount)
        return False

class TransactionLog:
    """Transaction logging system for bank operations."""
    
    def __init__(self):
        self.transactions: List[Dict] = []
    
    def log_transaction(
        self,
        transaction_type: str,
        amount: float,
        account_id: str,
        success: bool
    ) -> None:
        """Log a transaction with its details."""
        self.transactions.append({
            "type": transaction_type,
            "amount": amount,
            "account_id": account_id,
            "success": success
        })
    
    def get_account_transactions(self, account_id: str) -> List[Dict]:
        """Get all transactions for a specific account."""
        return [t for t in self.transactions if t["account_id"] == account_id]

class Bank:
    """Bank class combining accounts and transaction logging."""
    
    def __init__(self):
        self.accounts: Dict[str, BankAccount] = {}
        self.transaction_log = TransactionLog()
    
    def create_account(
        self,
        account_id: str,
        initial_balance: float = 0.0,
        overdraft_limit: float = 0.0
    ) -> bool:
        """Create a new bank account."""
        if account_id in self.accounts:
            return False
        
        self.accounts[account_id] = BankAccount(
            account_id,
            initial_balance,
            overdraft_limit
        )
        return True
    
    def get_account(self, account_id: str) -> Optional[BankAccount]:
        """Get account by ID."""
        return self.accounts.get(account_id)

# Property-based tests using hypothesis
class TestBankProperties:
    """Property-based tests for bank operations."""
    
    @given(
        balance=st.floats(min_value=0, max_value=1e6),
        amount=st.floats(min_value=0, max_value=1e6)
    )
    def test_deposit_withdrawal_property(self, balance: float, amount: float):
        """Test that deposit followed by withdrawal returns to original balance."""
        account = BankAccount("test", balance)
        original_balance = account.balance
        
        account.deposit(amount)
        account.withdraw(amount)
        
        # Account for floating point precision
        assert abs(account.balance - original_balance) < 1e-6
    
    @given(
        amount=st.floats(min_value=0, max_value=1e6),
        overdraft=st.floats(min_value=0, max_value=1e4)
    )
    def test_overdraft_property(self, amount: float, overdraft: float):
        """Test that accounts respect their overdraft limits."""
        account = BankAccount("test", 0.0, overdraft)
        success = account.withdraw(amount)
        
        if success:
            assert account.balance >= -account.overdraft_limit
        else:
            assert account.balance == 0.0

# Traditional tests with coverage analysis
class TestBank:
    """Traditional tests for bank operations."""
    
    @pytest.fixture
    def bank(self) -> Bank:
        """Fixture providing a bank instance."""
        return Bank()
    
    def test_create_account(self, bank: Bank):
        """Test account creation."""
        assert bank.create_account("1", 100.0)
        assert not bank.create_account("1", 200.0)  # Duplicate ID
        
        account = bank.get_account("1")
        assert account is not None
        assert account.balance == 100.0
    
    def test_transfer_between_accounts(self, bank: Bank):
        """Test money transfer between accounts."""
        bank.create_account("1", 100.0)
        bank.create_account("2", 50.0)
        
        account1 = bank.get_account("1")
        account2 = bank.get_account("2")
        
        assert account1 is not None and account2 is not None
        assert account1.transfer(account2, 30.0)
        
        assert account1.balance == 70.0
        assert account2.balance == 80.0
    
    def test_transaction_logging(self, bank: Bank):
        """Test transaction logging functionality."""
        bank.create_account("1", 100.0)
        account = bank.get_account("1")
        assert account is not None
        
        bank.transaction_log.log_transaction("deposit", 50.0, "1", True)
        bank.transaction_log.log_transaction("withdraw", 30.0, "1", True)
        
        transactions = bank.transaction_log.get_account_transactions("1")
        assert len(transactions) == 2
        assert transactions[0]["amount"] == 50.0
        assert transactions[1]["amount"] == 30.0

if __name__ == '__main__':
    # Run tests with coverage analysis
    cov = coverage.Coverage()
    cov.start()
    
    pytest.main([__file__])
    
    cov.stop()
    cov.save()
    
    # Generate coverage report
    cov.report() 