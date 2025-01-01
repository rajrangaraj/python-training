Feature: Shopping Cart
    As a customer
    I want to manage items in my shopping cart
    So that I can purchase products

    Scenario: Add single item to cart
        Given an empty shopping cart
        And a product catalog containing a book
        When I add 1 book to the cart
        Then the cart should contain 1 items
        And the total price should be $29.99

    Scenario: Add multiple items to cart
        Given an empty shopping cart
        And a product catalog containing a mouse
        When I add 3 mouse to the cart
        Then the cart should contain 3 items
        And the total price should be $59.97

    Scenario: Remove item from cart
        Given an empty shopping cart
        And a product catalog containing a book
        When I add 1 book to the cart
        And I remove the book from the cart
        Then the cart should contain 0 items
        And the total price should be $0.00

    Scenario: Add item with insufficient stock
        Given an empty shopping cart
        And a product catalog containing a laptop
        When I try to add 3 laptop to the cart
        Then I should get a "Not enough stock" error

    Scenario: Add item with invalid quantity
        Given an empty shopping cart
        And a product catalog containing a book
        When I try to add 0 book to the cart
        Then I should get a "Quantity must be positive" error 