import sqlite3
import pandas as pd
# Load the cleaned datasets
clean_orders = pd.read_csv("data/clean/orders.csv")
clean_customers = pd.read_csv("data/clean/customers.csv")
# Convert order timestamps into datetime format
clean_orders["order_ts"] = pd.to_datetime(clean_orders["order_ts"])
# Keep only completed orders for revenue analysis
completed_orders = clean_orders[
    clean_orders["status"] == "completed"
].copy()
# 1. PANDAS CROSS-CHECK: REVENUE BY CITY AND CATEGORY
# Connect completed orders with customer city information
orders_with_city = completed_orders.merge(
    clean_customers[["customer_id", "city"]],
    on="customer_id"
)
# Add product category information to the order records
orders_with_city_category = orders_with_city.merge(
    pd.read_csv("data/clean/products.csv")[["product_id", "category"]],
    on="product_id"
)
# Create a revenue summary by city and product category
revenue_pivot = pd.pivot_table(
    orders_with_city_category,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    fill_value=0
)
print("\n========== REVENUE PIVOT ==========")
print(revenue_pivot)
# 2. CITY-WISE REVENUE ANALYSIS
# Calculate total revenue, average order value, and order count for each city
city_revenue_summary = orders_with_city.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    completed_orders=("order_id", "count")
)
print("\n========== CITY ANALYSIS ==========")
print(city_revenue_summary)
# 3. COMPLETED ORDER AMOUNT STATISTICS
# Review the distribution of completed order amounts
print("\n========== ORDER AMOUNT STATISTICS ==========")
print(completed_orders["order_amount"].describe())
print("\nMean:", completed_orders["order_amount"].mean())
print("Median:", completed_orders["order_amount"].median())
print("Skewness:", completed_orders["order_amount"].skew())
# 4. OUTLIER DETECTION USING IQR
# Calculate the first and third quartiles of order amounts
first_quartile = completed_orders["order_amount"].quantile(0.25)
third_quartile = completed_orders["order_amount"].quantile(0.75)
# Calculate the interquartile range
interquartile_range = third_quartile - first_quartile
# Set the lower and upper boundaries for detecting outliers
lower_boundary = first_quartile - 1.5 * interquartile_range
upper_boundary = third_quartile + 1.5 * interquartile_range
# Identify orders falling outside the IQR boundaries
outlier_orders = completed_orders[
    (completed_orders["order_amount"] < lower_boundary) |
    (completed_orders["order_amount"] > upper_boundary)
]
print("\n========== OUTLIER ANALYSIS ==========")
print("Q1:", first_quartile)
print("Q3:", third_quartile)
print("IQR:", interquartile_range)
print("Lower Fence:", lower_boundary)
print("Upper Fence:", upper_boundary)
print("Outlier Count:", len(outlier_orders))
# 5. CORRELATION ANALYSIS
# Examine relationships between order amount, discount, quantity, and delivery time
correlation_matrix = completed_orders[
    ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
].corr()
print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)
# Count outliers separately below and above the IQR boundaries
lower_outlier_orders = completed_orders[
    completed_orders["order_amount"] < lower_boundary
]
upper_outlier_orders = completed_orders[
    completed_orders["order_amount"] > upper_boundary
]
print("Lower outliers:", len(lower_outlier_orders))
print("Upper outliers:", len(upper_outlier_orders))
# Display the Pandas pivot table's overall revenue total
# This will be compared against the SQL revenue total
print("Pandas grand total:", revenue_pivot.loc["All", "All"])
# Connect to the Instamart SQLite database
connection = sqlite3.connect("data/instamart.db")
# Calculate total revenue directly from the SQL orders table
sql_revenue = pd.read_sql_query(
    """
    SELECT SUM(order_amount) AS total_revenue
    FROM orders
    WHERE status = 'completed'
    """,
    connection
)["total_revenue"].iloc[0]
# Close the database connection after retrieving the result
connection.close()
print("SQL revenue:", sql_revenue)
# Confirm that Pandas and SQL produce the same revenue total
assert round(revenue_pivot.loc["All", "All"], 2) == round(sql_revenue, 2)
# Identify the strongest correlation between two different variables
# Remove diagonal values so each variable is not compared with itself
absolute_correlation = correlation_matrix.abs().copy()
# Mask self-correlations without modifying read-only arrays
absolute_correlation = absolute_correlation.where(~pd.DataFrame(
    absolute_correlation.index.to_numpy()[:, None] == absolute_correlation.columns.to_numpy(),
    index=absolute_correlation.index,
    columns=absolute_correlation.columns
))
strongest_correlation_pair = absolute_correlation.unstack().idxmax()
print("Strongest correlation pair:", strongest_correlation_pair)
print("Correlation value:", correlation_matrix.loc[
    strongest_correlation_pair[0], strongest_correlation_pair[1]
])

