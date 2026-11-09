import pandas as pd
import numpy as np
# Load the cleaned datasets
clean_orders = pd.read_csv("data/clean/orders.csv")
clean_customers = pd.read_csv("data/clean/customers.csv")
# Convert order timestamps into datetime format
clean_orders["order_ts"] = pd.to_datetime(clean_orders["order_ts"])
# Set the reference date for RFM calculations
rfm_snapshot_date = pd.Timestamp("2025-10-01 00:00:00")
# Keep only completed orders for customer analysis
completed_orders = clean_orders[

    clean_orders["status"] == "completed"
].copy()
# Calculate days since each customer's last order, frequency, and monetary value for each customer
rfm_summary = completed_orders.groupby("customer_id").agg(

    last_order_ts=("order_ts", "max"),

    frequency=("order_id", "nunique"),

    monetary=("order_amount", "sum")

).reset_index()
# Calculate days since each customer's last order
rfm_summary["recency"] = (

    rfm_snapshot_date - rfm_summary["last_order_ts"]
).dt.days
# Remove last order timestamp
rfm_summary = rfm_summary[

    ["customer_id", "recency", "frequency", "monetary"]
]
# Display RFM results
print("\n========== RFM TABLE ==========")
print(rfm_summary.head())
print("\n========== RFM SUMMARY ==========")
print(rfm_summary[["recency", "frequency", "monetary"]].describe())
print("\nCustomer count:", len(rfm_summary))
print("Recency median:", rfm_summary["recency"].median())
print("Recency minimum:", rfm_summary["recency"].min())
print("Recency maximum:", rfm_summary["recency"].max())
print("Frequency mean:", rfm_summary["frequency"].mean())
print("Frequency maximum:", rfm_summary["frequency"].max())
print("Monetary mean:", rfm_summary["monetary"].mean())
print("Monetary maximum:", rfm_summary["monetary"].max())
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Select RFM features
rfm_input_features = rfm_summary[

    ["recency", "frequency", "monetary"]

]
# Scale RFM metrics so they are comparable
scaler = StandardScaler()
scaled_rfm_features = scaler.fit_transform(rfm_input_features)
# Store clustering results
kmeans_results = []
# Test K values from 2 to 8
for cluster_count in range(2, 9):



    kmeans = KMeans(

        n_clusters=cluster_count,

        n_init=10,

        random_state=42

    )



    cluster_labels = kmeans.fit_predict(scaled_rfm_features)



    cluster_inertia = kmeans.inertia_



    silhouette_score_value = silhouette_score(

        scaled_rfm_features,

        cluster_labels

    )



    kmeans_results.append({

        "K": cluster_count,

        "Inertia": cluster_inertia,

        "Silhouette": silhouette_score_value

    })
# Display results
kmeans_results_df = pd.DataFrame(kmeans_results)
print("\n========== K-MEANS RESULTS ==========")
print(kmeans_results_df.round(3))
from sklearn.decomposition import PCA
import os
# TASK 3: FINAL CLUSTERING WITH K = 4
final_customer_kmeans = KMeans(

    n_clusters=4,

    n_init=10,

    random_state=42
)
rfm_summary["segment_id"] = final_customer_kmeans.fit_predict(scaled_rfm_features)
# Summarise customer behaviour within each segment
customer_segment_profile = rfm_summary.groupby("segment_id").agg(

    customer_count=("customer_id", "count"),

    mean_recency=("recency", "mean"),

    mean_frequency=("frequency", "mean"),

    mean_monetary=("monetary", "mean")
)
print("\n========== SEGMENT PROFILE ==========")
print(customer_segment_profile.round(2))
# PCA: REDUCE RFM TO TWO DIMENSIONS
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(scaled_rfm_features)
print("\n========== PCA RESULTS ==========")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
# NAME SEGMENTS
# Find the segment with the highest average spending
premium_segment_id = customer_segment_profile["mean_monetary"].idxmax()
# Assign business names based on segment profiles
customer_segment_names = {}
for customer_segment_id, profile in customer_segment_profile.iterrows():



    if customer_segment_id == premium_segment_id:

        customer_segment_names[customer_segment_id] = "Premium Customers"



    elif profile["mean_recency"] > rfm_summary["recency"].mean():

        customer_segment_names[customer_segment_id] = "Inactive Customers"



    elif profile["mean_frequency"] > rfm_summary["frequency"].mean():

        customer_segment_names[customer_segment_id] = "Frequent Customers"



    else:

        customer_segment_names[customer_segment_id] = "Occasional Customers"
# Assign segment names
rfm_summary["segment_name"] = rfm_summary["segment_id"].map(customer_segment_names)
# Flag premium customers
rfm_summary["is_premium"] = (

    rfm_summary["segment_id"] == premium_segment_id
)
print("\nPremium segment ID:", premium_segment_id)
# SAVE CUSTOMER SEGMENTS
os.makedirs("outputs", exist_ok=True)
customer_segments = rfm_summary[

    [

        "customer_id",

        "recency",

        "frequency",

        "monetary",

        "segment_id",

        "segment_name",

        "is_premium"

    ]

]
customer_segments.to_csv(

    "outputs/customer_segments.csv",

    index=False

)
print("\nCustomer segments saved successfully.")
print(customer_segments.head())
# TASK 4: LAPSE PREDICTION DATASET
# Define cutoff date and lapse observation period
lapse_cutoff_date = pd.Timestamp("2025-09-01")
lapse_observation_end = pd.Timestamp("2025-10-01")
# Completed orders before cutoff
orders_before_cutoff = completed_orders[

    completed_orders["order_ts"] < lapse_cutoff_date
].copy()
# Customers who purchased before cutoff
pre_cutoff_customer_ids = orders_before_cutoff["customer_id"].unique()
# Customers who purchased during September
orders_during_september = completed_orders[

    (completed_orders["order_ts"] >= lapse_cutoff_date) &

    (completed_orders["order_ts"] < lapse_observation_end)
]
september_customer_ids = orders_during_september["customer_id"].unique()
# Create cohort dataset
lapse_labels = pd.DataFrame({

    "customer_id": pre_cutoff_customer_ids

})
# Mark customers with no September purchase as lapsed
lapse_labels["lapsed"] = (

    ~lapse_labels["customer_id"].isin(september_customer_ids)

).astype(int)
# Display results
print("\n========== LAPSE COHORT RESULTS ==========")
print("Cohort size:", len(lapse_labels))
print("Lapsed customers:", lapse_labels["lapsed"].sum())
print("Lapse percentage:", round(lapse_labels["lapsed"].mean() * 100, 2))
print("\nTarget distribution:")
print(lapse_labels["lapsed"].value_counts())
# TASK 4: CUSTOMER FEATURES
# Recency, frequency and monetary before cutoff
lapse_customer_features = orders_before_cutoff.groupby("customer_id").agg(

    last_order=("order_ts", "max"),

    frequency=("order_id", "nunique"),

    monetary=("order_amount", "sum"),

    avg_discount=("discount_pct", "mean"),

    avg_delivery=("delivery_minutes", "mean")
).reset_index()
# Recency calculated from cutoff date
lapse_customer_features["recency"] = (

    lapse_cutoff_date - lapse_customer_features["last_order"]

).dt.days
# Cancellation rate before cutoff
all_orders_before_cutoff = clean_orders[clean_orders["order_ts"] < lapse_cutoff_date].copy()
customer_cancellation_summary = all_orders_before_cutoff.groupby("customer_id").agg(

    total_orders=("order_id", "nunique"),

    cancelled_orders=("status", lambda x: (x == "cancelled").sum())

).reset_index()
customer_cancellation_summary["cancellation_rate"] = (

    customer_cancellation_summary["cancelled_orders"] /

    customer_cancellation_summary["total_orders"]
)
# Merge cancellation rate
lapse_customer_features = lapse_customer_features.merge(

    customer_cancellation_summary[["customer_id", "cancellation_rate"]],

    on="customer_id",

    how="left"
)
# Customer details
lapse_customer_features = lapse_customer_features.merge(

    clean_customers[["customer_id", "city", "age", "gender", "signup_date"]],

    on="customer_id",

    how="left"
)
# Customer tenure as of cutoff
lapse_customer_features["signup_date"] = pd.to_datetime(

    lapse_customer_features["signup_date"]
)
lapse_customer_features["tenure_days"] = (

    lapse_cutoff_date - lapse_customer_features["signup_date"]

).dt.days
# Load products
products = pd.read_csv("data/clean/products.csv")
# Add product category to pre-cutoff orders
orders_before_cutoff = orders_before_cutoff.merge(

    products[["product_id", "category"]],

    on="product_id",

    how="left"
)
# Find most frequently purchased category for each customer
customer_favorite_category = orders_before_cutoff.groupby("customer_id")[

    "category"

].agg(lambda x: x.mode().iloc[0]).reset_index()
customer_favorite_category.columns = ["customer_id", "favorite_category"]
# Merge customer features and favorite category
lapse_customer_features = lapse_customer_features.merge(

    customer_favorite_category,

    on="customer_id",

    how="left"
)
# Merge target label
lapse_model_dataset = lapse_customer_features.merge(

    lapse_labels[["customer_id", "lapsed"]],

    on="customer_id",

    how="inner"
)
print("\nLAPSE DATASET COLUMNS:")
print(lapse_model_dataset.columns.tolist())
# TASK 4: TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
# Define features and target
model_features = lapse_model_dataset.drop(

    columns=["customer_id", "last_order", "signup_date", "lapsed"]

)
lapse_target = lapse_model_dataset["lapsed"]
# Split dataset
features_train, features_test, target_train, target_test = train_test_split(

    model_features,

    lapse_target,

    test_size=0.20,

    random_state=42,

    stratify=lapse_target
)
print("\n========== TRAIN-TEST SPLIT ==========")
print("Training rows:", len(features_train))
print("Testing rows:", len(features_test))
print("\nTraining target distribution:")
print(target_train.value_counts())
print("\nTesting target distribution:")
print(target_test.value_counts())
# TASK 5: PREPROCESSING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
# Define numerical and categorical columns
numerical_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "tenure_days",
    "age"
]
categorical_columns = [
    "city",
    "gender",
    "favorite_category"
]
# Apply scaling to numerical features and encoding to categorical features
feature_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         categorical_columns)
    ]
)
# Learn preprocessing values from training data only
X_train_processed = feature_preprocessor.fit_transform(features_train)
# Apply learned preprocessing to test data
X_test_processed = feature_preprocessor.transform(features_test)
X_test_processed = feature_preprocessor.transform(features_test)
print("\n========== PREPROCESSING RESULTS ==========")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)
print("\nMissing values in X_train:", features_train.isnull().sum().sum())
print("Missing values in X_test:", features_test.isnull().sum().sum())
# Compare scaler training mean with full-cohort mean
scaler_frequency_mean = feature_preprocessor.named_transformers_["num"].mean_[1]
full_cohort_frequency_mean = lapse_model_dataset["frequency"].mean()
print("\nScaler frequency mean:", scaler_frequency_mean)
print("Full cohort frequency mean:", full_cohort_frequency_mean)
# TASK 6: LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Logistic Regression pipeline
logistic_pipeline = Pipeline([

    ("preprocessor", feature_preprocessor),

    ("model", LogisticRegression(max_iter=1000))
])
# Hyperparameter grid
logistic_parameter_grid = {

    "model__C": [0.01, 0.1, 1, 10, 100]

}
# Grid Search
logistic_grid_search = GridSearchCV(

    logistic_pipeline,

    logistic_parameter_grid,

    cv=5,

    scoring="f1"
)
logistic_grid_search.fit(features_train, target_train)
print("\n========== LOGISTIC REGRESSION ==========")
print("Best parameters:", logistic_grid_search.best_params_)
print("Best CV F1:", logistic_grid_search.best_score_)
# TASK 6: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
random_forest_pipeline = Pipeline([

    ("preprocessor", feature_preprocessor),

    ("model", RandomForestClassifier(

        n_estimators=100,

        random_state=42

    ))
])
random_forest_parameter_grid = {

    "model__max_depth": [3, 5, 8],

    "model__min_samples_leaf": [5, 10, 20]

}
random_forest_grid_search = GridSearchCV(

    random_forest_pipeline,

    random_forest_parameter_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1
)
random_forest_grid_search.fit(features_train, target_train)
print("\n========== RANDOM FOREST ==========")
print("Best parameters:", random_forest_grid_search.best_params_)
print("Best CV F1:", random_forest_grid_search.best_score_)
# TASK 6: MODEL EVALUATION
from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score,

    f1_score
)
# Best models
best_logistic_model = logistic_grid_search.best_estimator_
best_random_forest_model = random_forest_grid_search.best_estimator_
# Predictions on untouched test data
logistic_predictions = best_logistic_model.predict(features_test)
random_forest_predictions = best_random_forest_model.predict(features_test)
# Baseline model: predict every customer as not lapsed (0)
baseline_model_predictions = np.zeros(len(target_test), dtype=int)
baseline_model_accuracy = accuracy_score(target_test, baseline_model_predictions)
baseline_lapsed_f1 = f1_score(target_test, baseline_model_predictions, zero_division=0)
print("\n========== BASELINE MODEL ==========")
print("Baseline Accuracy:", baseline_model_accuracy)
print("Baseline Lapsed-Class F1:", baseline_lapsed_f1)
# Logistic Regression results
print("\n========== LOGISTIC REGRESSION TEST ==========")
print("Accuracy:", accuracy_score(target_test, logistic_predictions))
print("Confusion Matrix:")
print(confusion_matrix(target_test, logistic_predictions))
print(classification_report(target_test, logistic_predictions, zero_division=0))
# Random Forest results
print("\n========== RANDOM FOREST TEST ==========")
print("Accuracy:", accuracy_score(target_test, random_forest_predictions))
print("Confusion Matrix:")
print(confusion_matrix(target_test, random_forest_predictions))
print(classification_report(target_test, random_forest_predictions, zero_division=0))
# Compare baseline and tuned models
print("\n========== MODEL COMPARISON ==========")
print("Baseline Accuracy:", baseline_model_accuracy)
print("Baseline F1:", baseline_lapsed_f1)
print("Logistic Regression F1:", f1_score(target_test, logistic_predictions))
print("Random Forest F1:", f1_score(target_test, random_forest_predictions))
# TASK 7: OVERSAMPLING
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score
# Combine training features and target
training_data = features_train.copy()
training_data["lapsed"] = target_train.values
# Separate majority and minority classes
majority_class_data = training_data[training_data["lapsed"] == 0]
lapsed_class_data = training_data[training_data["lapsed"] == 1]
# Calculate how many extra lapsed rows are needed
additional_lapsed_samples = len(majority_class_data) - len(lapsed_class_data)
# Randomly duplicate only the required number of lapsed rows
extra_lapsed_rows = resample(

    lapsed_class_data,

    replace=True,

    n_samples=additional_lapsed_samples,

    random_state=42

)
# Keep all original rows and add the extra lapsed rows
balanced_training_data = pd.concat(

    [majority_class_data, lapsed_class_data, extra_lapsed_rows]

)
X_train_balanced = balanced_training_data.drop(columns=["lapsed"])
y_train_balanced = balanced_training_data["lapsed"]
print("\n========== OVERSAMPLING RESULTS ==========")
print(y_train_balanced.value_counts())
# Train Logistic Regression on balanced data

oversampled_logistic_model = Pipeline([

    ("preprocessor", feature_preprocessor),

    ("model", LogisticRegression(

        C=logistic_grid_search.best_params_["model__C"],

        max_iter=1000

    ))

])
oversampled_logistic_model.fit(X_train_balanced, y_train_balanced)
# Evaluate on original test data
oversampled_model_predictions = oversampled_logistic_model.predict(features_test)
print("\n========== OVERSAMPLING COMPARISON ==========")
print("\nOriginal Tuned Logistic Regression:")
print("Precision:", precision_score(target_test, logistic_predictions))
print("Recall:", recall_score(target_test, logistic_predictions))
print("F1:", f1_score(target_test, logistic_predictions))
print("\nOversampled Logistic Regression:")
print("Precision:", precision_score(target_test, oversampled_model_predictions))
print("Recall:", recall_score(target_test, oversampled_model_predictions))
print("F1:", f1_score(target_test, oversampled_model_predictions))
# TASK 8: FEATURE IMPORTANCE
# Random Forest feature importance
random_forest_model = best_random_forest_model.named_steps["model"]
random_forest_feature_names = best_random_forest_model.named_steps[

    "preprocessor"
].get_feature_names_out()
random_forest_importance = pd.DataFrame({

    "feature": random_forest_feature_names,

    "importance": random_forest_model.feature_importances_

}).sort_values("importance", ascending=False)
print("\n========== TOP 5 RF FEATURES ==========")
print(random_forest_importance.head(5))
# Logistic Regression coefficients
logistic_model = best_logistic_model.named_steps["model"]
logistic_feature_names = best_logistic_model.named_steps[

    "preprocessor"
].get_feature_names_out()
logistic_coefficients = pd.DataFrame({

    "feature": logistic_feature_names,

    "coefficient": logistic_model.coef_[0]

})
# Select top 5 Random Forest features
top_five_features = random_forest_importance.head(5).copy()
# Match RF features with LR coefficients
top_five_features = top_five_features.merge(

    logistic_coefficients,

    on="feature",

    how="left",

    validate="one_to_one"
)
# Identify LR coefficient direction
top_five_features["lr_sign"] = np.where(

    top_five_features["coefficient"] > 0,

    "Positive",

    np.where(

        top_five_features["coefficient"] < 0,

        "Negative",

        "Zero"

    )

)
# Feature-specific business descriptions
feature_business_meanings = {

    "recency": "Days since last order",

    "frequency": "Number of orders",

    "monetary": "Total customer spending",

    "avg_discount": "Average discount received",

    "cancellation_rate": "Proportion of cancelled orders",

    "avg_delivery": "Average delivery time",

    "tenure_days": "Days since signup",

    "age": "Customer age"

}
# Generate one business interpretation per feature
def explain_feature(feature_row):

    feature_name = feature_row["feature"].split("__")[-1]

    feature_meaning = feature_business_meanings.get(feature_name, feature_name)



    if pd.isna(feature_row["coefficient"]):

        return f"{feature_meaning}: LR coefficient unavailable."



    if feature_row["coefficient"] > 0:

        risk_direction = "higher predicted lapse risk"

    elif feature_row["coefficient"] < 0:

        risk_direction = "lower predicted lapse risk"

    else:

        risk_direction = "no directional association with lapse risk"



    return (

        f"{feature_meaning}: a higher value is associated with "

        f"{risk_direction}."

    )
top_five_features["business_reading"] = top_five_features.apply(

    explain_feature,

    axis=1
)
# Display final feature importance comparison
print("\n========== TOP 5 RF FEATURES WITH LR SIGNS ==========")
print(

    top_five_features[

        [

            "feature",

            "importance",

            "coefficient",

            "lr_sign",

            "business_reading"

        ]

    ].to_string(index=False)

)
# TASK 8: SAVE AND RELOAD MODEL
import joblib
import os
os.makedirs("outputs", exist_ok=True)
# Select the final model based on tuned model comparison
final_selected_model = best_logistic_model
# Generate predictions before saving
predictions_before_saving = final_selected_model.predict(features_test)
# Save the selected model
joblib.dump(final_selected_model, "outputs/lapse_model.joblib")
# Reload the saved model
reloaded_selected_model = joblib.load("outputs/lapse_model.joblib")
# Generate predictions after reloading
predictions_after_reloading = reloaded_selected_model.predict(features_test)
# Verify predictions are identical
reload_predictions_match = np.array_equal(

    predictions_before_saving,

    predictions_after_reloading
)
print("\n========== MODEL RELOAD VERIFICATION ==========")
print("Selected Model: Tuned Logistic Regression")
print("Predictions identical:", reload_predictions_match)
assert reload_predictions_match, "Model predictions changed after reloading!"
# TASK 8: RETENTION LIST
# Predict lapse probability for test customers
predicted_lapse_probabilities = reloaded_selected_model.predict_proba(features_test)[:, 1]
# Create test customer results
premium_retention_list = pd.DataFrame({

    "customer_id": lapse_model_dataset.loc[features_test.index, "customer_id"].values,

    "lapse_probability": predicted_lapse_probabilities,

    "actually_lapsed": target_test.values

})
# Load premium segment customers
customer_segments = pd.read_csv("outputs/customer_segments.csv")
premium_customer_ids = customer_segments[

    customer_segments["is_premium"] == True

]["customer_id"]
# Filter premium customers from test set
premium_retention_list = premium_retention_list[

    premium_retention_list["customer_id"].isin(premium_customer_ids)

].copy()
# Rank premium customers by highest lapse probability
premium_retention_list = premium_retention_list.sort_values(

    "lapse_probability",

    ascending=False
)
# Select only the top 10 premium customers
premium_retention_list = premium_retention_list.head(10).copy()
# Save the top 10 retention list
premium_retention_list.to_csv(

    "outputs/retention_list.csv",

    index=False
)
print("\n========== TOP 10 PREMIUM RETENTION LIST ==========")
print("Customers selected:", len(premium_retention_list))
print("Actually lapsed:", premium_retention_list["actually_lapsed"].sum())
print("\nTop 10 Premium Customers:")
print(premium_retention_list.to_string(index=False))
print("\nTop 10 retention customers:")
print(premium_retention_list.head(10))
