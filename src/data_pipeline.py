import pandas as pd
import os
# TASK 2: AUDIT RAW DATA
# Load the raw datasets
data_path = "data/raw/"
customers_raw = pd.read_csv(data_path + "customers.csv")
products_raw = pd.read_csv(data_path + "products.csv")
orders_raw = pd.read_csv(data_path + "orders.csv")
raw_datasets = {
    "Customers": customers_raw,
    "Products": products_raw,
    "Orders": orders_raw
}
# Check dataset structure, data types, and missing values
for dataset_name, dataset in raw_datasets.items():
    print("\n==========", dataset_name, "==========")
    print("Shape:", dataset.shape)

    print("\nData Types:")
    print(dataset.dtypes)

    missing_values = dataset.isnull().sum()
    missing_percentages = (missing_values / len(dataset)) * 100

    missing_summary = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage": missing_percentages
    })

    print("\nMissing Values:")
    print(missing_summary)
# Check order-related data quality issues
print("\nExact duplicate rows:", orders_raw.duplicated().sum())
print("Pending orders:", (orders_raw["status"] == "pending").sum())
print("Distinct order IDs:", orders_raw["order_id"].nunique())

numeric_order_amounts = pd.to_numeric(
    orders_raw["order_amount"], errors="coerce"
)
print("Non-numeric or blank order amounts:", numeric_order_amounts.isna().sum())
print("Negative delivery minutes:", (orders_raw["delivery_minutes"] < 0).sum())
print("Blank delivery minutes:", orders_raw["delivery_minutes"].isna().sum())
# Check customer-related data quality issues
print("\nBlank ages:", customers_raw["age"].isna().sum())
invalid_customer_ages = (
    (customers_raw["age"] < 10) | (customers_raw["age"] > 100)
)
print("Ages outside 10-100:", invalid_customer_ages.sum())
print("Distinct raw city spellings:", customers_raw["city"].nunique())
print(
    "Distinct non-blank raw gender spellings:",
    customers_raw["gender"].dropna().nunique()
)
print("Blank genders:", customers_raw["gender"].isna().sum())
# TASK 3: CLEAN CUSTOMERS
# Inspect customer records and identify duplicates
print(customers_raw.head(10))
customers_raw.info()
print("Duplicate rows:", customers_raw.duplicated().sum())
print("Duplicate customer IDs:", customers_raw["customer_id"].duplicated().sum())
print(customers_raw["age"].describe())
invalid_age_records = customers_raw[
    (customers_raw["age"] < 10) | (customers_raw["age"] > 100)
]
print("Invalid age count:", len(invalid_age_records))
print(invalid_age_records[["customer_id", "age"]])
# Create a separate copy for customer cleaning
customers_cleaned = customers_raw.copy()
# Standardize city names before filling missing ages
customers_cleaned["city"] = (
    customers_cleaned["city"].str.strip().str.title()
)
city_values_changed = (
    customers_raw["city"] != customers_cleaned["city"]
).sum()
print("\nCity values standardized:", city_values_changed)
print("\nCities after standardization:")
print(customers_cleaned["city"].value_counts())
print("\nNumber of unique cities:", customers_cleaned["city"].nunique())
# Standardize gender values into consistent categories
gender_mapping = {
    "female": "female",
    "f": "female",
    "male": "male",
    "m": "male"
}
customers_cleaned["gender"] = (
    customers_cleaned["gender"]
    .str.strip()
    .str.lower()
    .map(gender_mapping)
    .fillna("unknown")
)
blank_gender_count = customers_raw["gender"].isna().sum()
gender_values_changed = (
    customers_raw["gender"].notna()
    & (customers_raw["gender"] != customers_cleaned["gender"])
).sum()
print("Gender values standardized (non-blank):", gender_values_changed)
print("Blank genders mapped to unknown:", blank_gender_count)
print("\nGender counts:")
print(customers_cleaned["gender"].value_counts())
# Replace invalid ages with missing values
# Fill missing ages using the median age of the customer's city
original_blank_age_count = customers_cleaned["age"].isna().sum()
invalid_age_count = (
    (customers_cleaned["age"] < 10) | (customers_cleaned["age"] > 100)
).sum()
print("\nOriginal blank ages:", original_blank_age_count)
print("Age outliers masked:", invalid_age_count)
customers_cleaned.loc[
    (customers_cleaned["age"] < 10) | (customers_cleaned["age"] > 100),
    "age"
] = pd.NA
missing_ages_before_filling = customers_cleaned["age"].isna().sum()
print("Missing ages before median imputation:", missing_ages_before_filling)
customers_cleaned["age"] = (
    customers_cleaned["age"].fillna(
        customers_cleaned.groupby("city")["age"].transform("median")
    )
)
missing_ages_after_filling = customers_cleaned["age"].isna().sum()
print(
    "Ages filled using city median:",
    missing_ages_before_filling - missing_ages_after_filling
)
print("Missing ages after median imputation:", missing_ages_after_filling)
# Convert signup dates into datetime format
customers_cleaned["signup_date"] = pd.to_datetime(
    customers_cleaned["signup_date"]
)
# Review the final age distribution
print("\nFinal Age Summary:")
print(customers_cleaned["age"].describe())
print("\nMinimum age:", customers_cleaned["age"].min())
print("Maximum age:", customers_cleaned["age"].max())
# TASK 4: CLEAN ORDERS
# Create a separate copy for order cleaning
clean_orders = orders_raw.copy()
# Step 1: Convert order timestamps into datetime format
clean_orders["order_ts"] = pd.to_datetime(clean_orders["order_ts"])
# Step 2: Remove exact duplicate order records
rows_before_removing_duplicates = len(clean_orders)
clean_orders = clean_orders.drop_duplicates()
print(
    "\nExact duplicate order rows removed:",
    rows_before_removing_duplicates - len(clean_orders)
)
# Step 3: Keep the latest record for each order_id
# Stable sorting preserves the original order when timestamps are tied
rows_before_removing_old_records = len(clean_orders)
clean_orders = (
    clean_orders
    .sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
)
print(
    "Older order_id records removed:",
    rows_before_removing_old_records - len(clean_orders)
)
# Step 4: Convert order amounts into numeric values
# Fill missing amounts using product price, quantity, and discount
clean_orders["order_amount"] = pd.to_numeric(
    clean_orders["order_amount"], errors="coerce"
)
clean_orders = clean_orders.merge(
    products_raw[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left"
)
missing_order_amounts = clean_orders["order_amount"].isnull()
print("Missing order amounts to fill:", missing_order_amounts.sum())
clean_orders.loc[missing_order_amounts, "order_amount"] = (
    clean_orders.loc[missing_order_amounts, "price_inr"]
    * clean_orders.loc[missing_order_amounts, "quantity"]
    * (1 - clean_orders.loc[missing_order_amounts, "discount_pct"] / 100)
).round(2)
# Step 5: Replace negative delivery times with missing values
negative_delivery_count = (
    clean_orders["delivery_minutes"] < 0
).sum()
print("Negative delivery minutes masked:", negative_delivery_count)
clean_orders.loc[
    clean_orders["delivery_minutes"] < 0, "delivery_minutes"
] = pd.NA
# Step 6: Fill missing delivery times for completed orders
# Use the median delivery time of the corresponding product category
completed_orders = clean_orders["status"] == "completed"
completed_missing_before_filling = clean_orders.loc[
    completed_orders, "delivery_minutes"
].isna().sum()
clean_orders.loc[completed_orders, "delivery_minutes"] = (
    clean_orders.loc[completed_orders, "delivery_minutes"]
    .fillna(
        clean_orders.loc[completed_orders]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
)
completed_missing_after_filling = clean_orders.loc[
    completed_orders, "delivery_minutes"
].isna().sum()
print(
    "Completed delivery times filled:",
    completed_missing_before_filling - completed_missing_after_filling
)
# Step 7: Clear delivery times for cancelled orders
cancelled_orders = clean_orders["status"] == "cancelled"
print(
    "Cancelled delivery values cleared:",
    clean_orders.loc[cancelled_orders, "delivery_minutes"].notna().sum()
)
clean_orders.loc[cancelled_orders, "delivery_minutes"] = pd.NA
# Remove temporary product information added during order cleaning
clean_orders = clean_orders.drop(columns=["category", "price_inr"])
# Check the final cleaned order records
print("\nCleaned orders shape:", clean_orders.shape)
print("\nOrder status counts:")
print(clean_orders["status"].value_counts())
print("\nUnique order IDs:", clean_orders["order_id"].nunique())
print(
    "\nMissing order amounts:",
    clean_orders["order_amount"].isnull().sum()
)
print(
    "\nMissing delivery minutes in completed orders:",
    clean_orders.loc[
        clean_orders["status"] == "completed", "delivery_minutes"
    ].isnull().sum()
)
print(
    "\nMissing delivery minutes in cancelled orders:",
    clean_orders.loc[
        clean_orders["status"] == "cancelled", "delivery_minutes"
    ].isnull().sum()
)
print(
    "\nCompleted order amount total INR:",
    clean_orders.loc[
        clean_orders["status"] == "completed", "order_amount"
    ].sum().round(2)
)
# Save the cleaned datasets as CSV files
os.makedirs("data/clean", exist_ok=True)
customers_cleaned.to_csv("data/clean/customers.csv", index=False)
print("Cleaned customers dataset saved successfully")
clean_orders.to_csv("data/clean/orders.csv", index=False)
print("Cleaned orders dataset saved successfully")
products_raw.to_csv("data/clean/products.csv", index=False)
print("Cleaned products dataset saved successfully")
# Reload the saved files and verify their shapes
clean_customers = pd.read_csv("data/clean/customers.csv")
clean_products = pd.read_csv("data/clean/products.csv")
clean_orders = pd.read_csv("data/clean/orders.csv")
print("\nCustomers shape:", clean_customers.shape)
print("Products shape:", clean_products.shape)
print("Orders shape:", clean_orders.shape)

