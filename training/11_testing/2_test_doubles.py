"""
Demonstration of test doubles (mocks, stubs, fakes, etc.) in Python using unittest.mock.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Optional
from datetime import datetime

# Classes to be tested
class EmailService:
    """Service for sending emails."""
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """
        Send an email (in real implementation, this would connect to an SMTP server).
        Returns True if successful, False otherwise.
        """
        # In reality, this would connect to an email server
        raise NotImplementedError("Real email service not implemented for demo")

class UserDatabase:
    """Database service for user operations."""
    
    def get_user_email(self, user_id: int) -> Optional[str]:
        """Get user's email address from database."""
        # In reality, this would query a database
        raise NotImplementedError("Real database not implemented for demo")
    
    def update_last_notified(self, user_id: int) -> None:
        """Update the last notification timestamp for a user."""
        # In reality, this would update a database record
        raise NotImplementedError("Real database not implemented for demo")

class NotificationService:
    """Service for sending user notifications."""
    
    def __init__(self, email_service: EmailService, user_db: UserDatabase):
        self.email_service = email_service
        self.user_db = user_db
    
    def notify_user(self, user_id: int, message: str) -> bool:
        """
        Notify a user with a message via email.
        Returns True if notification was sent successfully.
        """
        email = self.user_db.get_user_email(user_id)
        if not email:
            return False
        
        success = self.email_service.send_email(
            to=email,
            subject="New Notification",
            body=message
        )
        
        if success:
            self.user_db.update_last_notified(user_id)
        
        return success

# Test class
class TestNotificationService(unittest.TestCase):
    """Test cases for NotificationService using test doubles."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock objects
        self.email_service = Mock(spec=EmailService)
        self.user_db = Mock(spec=UserDatabase)
        
        # Create the service with mock dependencies
        self.notification_service = NotificationService(
            self.email_service,
            self.user_db
        )
    
    def test_notify_user_success(self):
        """Test successful notification."""
        # Setup mock behavior
        self.user_db.get_user_email.return_value = "user@example.com"
        self.email_service.send_email.return_value = True
        
        # Call the method under test
        result = self.notification_service.notify_user(1, "Test message")
        
        # Assertions
        self.assertTrue(result)
        self.user_db.get_user_email.assert_called_once_with(1)
        self.email_service.send_email.assert_called_once_with(
            to="user@example.com",
            subject="New Notification",
            body="Test message"
        )
        self.user_db.update_last_notified.assert_called_once_with(1)
    
    def test_notify_user_no_email(self):
        """Test notification when user email is not found."""
        # Setup mock behavior
        self.user_db.get_user_email.return_value = None
        
        # Call the method under test
        result = self.notification_service.notify_user(1, "Test message")
        
        # Assertions
        self.assertFalse(result)
        self.user_db.get_user_email.assert_called_once_with(1)
        self.email_service.send_email.assert_not_called()
        self.user_db.update_last_notified.assert_not_called()
    
    def test_notify_user_email_failure(self):
        """Test notification when email sending fails."""
        # Setup mock behavior
        self.user_db.get_user_email.return_value = "user@example.com"
        self.email_service.send_email.return_value = False
        
        # Call the method under test
        result = self.notification_service.notify_user(1, "Test message")
        
        # Assertions
        self.assertFalse(result)
        self.user_db.update_last_notified.assert_not_called()

    @patch('datetime.datetime')
    def test_with_datetime_mock(self, mock_datetime):
        """Test using patch decorator to mock datetime."""
        # Setup mock datetime
        mock_now = datetime(2024, 1, 1, 12, 0)
        mock_datetime.now.return_value = mock_now
        
        # Use datetime.now() somewhere
        current_time = datetime.now()
        self.assertEqual(current_time, mock_now)

    def test_mock_multiple_calls(self):
        """Test mocking multiple calls with different returns."""
        # Setup mock with multiple return values
        mock = Mock()
        mock.side_effect = [1, 2, 3]
        
        # Each call returns the next value
        self.assertEqual(mock(), 1)
        self.assertEqual(mock(), 2)
        self.assertEqual(mock(), 3)
        
        # Verify call count
        self.assertEqual(mock.call_count, 3)

    def test_mock_with_magic(self):
        """Test using MagicMock for special methods."""
        # MagicMock automatically implements magic methods
        magic_mock = MagicMock()
        
        # Can be used as an iterator
        for _ in magic_mock:
            break
        
        # Can be called like a function
        magic_mock()
        
        # Can be used with len()
        len(magic_mock)
        
        # All these operations are recorded
        self.assertTrue(magic_mock.__iter__.called)
        self.assertTrue(magic_mock.__call__.called)
        self.assertTrue(magic_mock.__len__.called)

if __name__ == '__main__':
    unittest.main() 