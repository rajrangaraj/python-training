"""
Demonstration of debugging techniques and logging in Python.
"""

import logging
import pdb
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)

class InventorySystem:
    def __init__(self):
        self.inventory = {}
        logging.info("Inventory system initialized")
    
    def add_item(self, item_id, name, quantity):
        """Add or update item in inventory with error checking."""
        try:
            # Input validation
            if not isinstance(item_id, str):
                raise ValueError("Item ID must be a string")
            if quantity < 0:
                raise ValueError("Quantity cannot be negative")
            
            # Add or update item
            if item_id in self.inventory:
                logging.info(f"Updating existing item: {item_id}")
            else:
                logging.info(f"Adding new item: {item_id}")
            
            self.inventory[item_id] = {
                'name': name,
                'quantity': quantity,
                'last_updated': datetime.now()
            }
            
            # Debug point example
            # pdb.set_trace()
            
            return True
        
        except Exception as e:
            logging.error(f"Error adding item {item_id}: {str(e)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def remove_item(self, item_id, quantity):
        """Remove quantity of item from inventory with error checking."""
        try:
            if item_id not in self.inventory:
                raise KeyError(f"Item {item_id} not found in inventory")
            
            current_quantity = self.inventory[item_id]['quantity']
            if quantity > current_quantity:
                raise ValueError(
                    f"Insufficient quantity. Have: {current_quantity}, "
                    f"Requested: {quantity}"
                )
            
            # Update quantity
            self.inventory[item_id]['quantity'] -= quantity
            logging.info(
                f"Removed {quantity} units of {item_id}. "
                f"New quantity: {self.inventory[item_id]['quantity']}"
            )
            
            # Remove item if quantity is 0
            if self.inventory[item_id]['quantity'] == 0:
                del self.inventory[item_id]
                logging.info(f"Item {item_id} removed from inventory")
            
            return True
        
        except Exception as e:
            logging.error(f"Error removing item {item_id}: {str(e)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def get_inventory_status(self):
        """Get current inventory status with debugging information."""
        try:
            status = {
                'total_items': len(self.inventory),
                'items': self.inventory,
                'last_checked': datetime.now()
            }
            
            logging.info("Inventory status checked")
            logging.debug(f"Current inventory: {self.inventory}")
            
            return status
        
        except Exception as e:
            logging.error(f"Error getting inventory status: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Create inventory system
    inventory = InventorySystem()
    
    # Add items
    print("Adding items to inventory:")
    items_to_add = [
        ("A123", "Laptop", 5),
        ("B456", "Mouse", 10),
        ("C789", "Keyboard", 7)
    ]
    
    for item_id, name, quantity in items_to_add:
        success = inventory.add_item(item_id, name, quantity)
        print(f"Added {name}: {'Success' if success else 'Failed'}")
    
    # Try to add invalid item
    print("\nTrying to add invalid item:")
    success = inventory.add_item("D123", "Monitor", -1)
    print(f"Added item with negative quantity: {'Success' if success else 'Failed'}")
    
    # Remove items
    print("\nRemoving items from inventory:")
    remove_operations = [
        ("A123", 2),  # Valid removal
        ("B456", 15), # Too many
        ("X999", 1)   # Non-existent item
    ]
    
    for item_id, quantity in remove_operations:
        success = inventory.remove_item(item_id, quantity)
        print(f"Removed {quantity} of {item_id}: {'Success' if success else 'Failed'}")
    
    # Check inventory status
    print("\nFinal inventory status:")
    status = inventory.get_inventory_status()
    if status:
        print(f"Total items: {status['total_items']}")
        for item_id, details in status['items'].items():
            print(f"{item_id}: {details['name']} - {details['quantity']} units") 