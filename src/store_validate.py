# ==========================================
# TASK 5: STORE AND VALIDATE
# ==========================================
# Import the libraries needed for data storage and validation
import pandas as pd
import sqlite3
import os
# Step 1: Create the folders needed to store cleaned data
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data", exist_ok=True)
# Step 2: Load the cleaned datasets
clean_customers = pd.read_csv("data/clean/customers.csv")
clean_products = pd.read_csv("data/clean/products.csv")
clean_orders = pd.read_csv("data/clean/orders.csv")
# Step 3: Format order timestamps consistently
clean_orders["order_ts"] = pd.to_datetime(
    clean_orders["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")
# Step 4: Save the cleaned datasets back as CSV files
clean_customers.to_csv("data/clean/customers.csv", index=False)
clean_products.to_csv("data/clean/products.csv", index=False)
clean_orders.to_csv("data/clean/orders.csv", index=False)
# Step 5: Connect to the SQLite database
database_connection = sqlite3.connect("data/instamart.db")
# Step 6: Store each cleaned dataset as a SQLite table
clean_customers.to_sql(
    "customers",
    database_connection,
    if_exists="replace",
    index=False
)
clean_products.to_sql(
    "products",
    database_connection,
    if_exists="replace",
    index=False
)
clean_orders.to_sql(
    "orders",
    database_connection,
    if_exists="replace",
    index=False
)
# ==========================================
# VALIDATION CHECKPOINTS
# ==========================================
# Check 1: Verify the number of customer records
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers",
    database_connection
)["count"][0] == 1200
# Check 2: Verify the number of product records
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM products",
    database_connection
)["count"][0] == 40
# Check 3: Verify the number of order records
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders",
    database_connection
)["count"][0] == 12309
# Check 4: Confirm that every order ID is unique
assert pd.read_sql_query(
    "SELECT COUNT(DISTINCT order_id) AS count FROM orders",
    database_connection
)["count"][0] == 12309
# Check 5: Verify the number of completed orders
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='completed'",
    database_connection
)["count"][0] == 11110
# Check 6: Verify the number of cancelled orders
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='cancelled'",
    database_connection
)["count"][0] == 1199
# Check 7: Confirm that no pending orders remain
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='pending'",
    database_connection
)["count"][0] == 0
# Check 8: Confirm that all orders have an amount
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE order_amount IS NULL",
    database_connection
)["count"][0] == 0
# Check 9: Confirm that completed orders have delivery times
assert pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM orders
    WHERE status='completed' AND delivery_minutes IS NULL
    """,
    database_connection
)["count"][0] == 0
# Check 10: Confirm that cancelled orders have no delivery times
assert pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM orders
    WHERE status='cancelled' AND delivery_minutes IS NULL
    """,
    database_connection
)["count"][0] == 1199
# Check 11: Verify total revenue from completed orders
completed_revenue = pd.read_sql_query(
    """
    SELECT SUM(order_amount) AS total
    FROM orders
    WHERE status='completed'
    """,
    database_connection
)["total"][0]
assert round(completed_revenue, 2) == 2998668.80
# Check 12: Confirm that all customers have an age
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers WHERE age IS NULL",
    database_connection
)["count"][0] == 0
# Check 13: Confirm that all customers have a gender value
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers WHERE gender IS NULL",
    database_connection
)["count"][0] == 0
# Check 14: Verify the minimum customer age
assert pd.read_sql_query(
    "SELECT MIN(age) AS minimum FROM customers",
    database_connection
)["minimum"][0] == 18
# Check 15: Verify the maximum customer age
assert pd.read_sql_query(
    "SELECT MAX(age) AS maximum FROM customers",
    database_connection
)["maximum"][0] == 60
# Step 7: Close the database connection
database_connection.close()
# Display the final validation success message
print("ALL CHECKPOINTS PASSED")
