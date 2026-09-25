from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("CustomerChurnDataValidation") \
    .getOrCreate()

print("\n======================================")
print("CUSTOMER CHURN DATA VALIDATION")
print("======================================")

project_path = "/content/drive/MyDrive/Customer Churn Prediction Pipeline"

cleaned_path = f"{project_path}/data/processed/cleaned_churn"

# Load cleaned data
df = spark.read.csv(
    cleaned_path,
    header=True,
    inferSchema=True
)

print("\nDataset loaded successfully!")
print("Rows:", df.count())
print("Columns:", len(df.columns))


# ======================================
# 1. REQUIRED COLUMNS
# ======================================

required_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in df.columns
]

print("\n======================================")
print("COLUMN VALIDATION")
print("======================================")

if missing_columns:
    print("Missing columns:", missing_columns)
    raise ValueError("Required columns are missing!")
else:
    print("All required columns are present.")


# ======================================
# 2. NULL VALUE CHECK
# ======================================

print("\n======================================")
print("NULL VALUE VALIDATION")
print("======================================")

null_columns = []

for column_name in df.columns:
    null_count = df.filter(
        col(column_name).isNull()
    ).count()

    if null_count > 0:
        null_columns.append(
            (column_name, null_count)
        )

if null_columns:
    print("Columns containing NULL values:")

    for column_name, count in null_columns:
        print(f"{column_name}: {count}")
else:
    print("No NULL values found.")


# ======================================
# 3. DUPLICATE CUSTOMER CHECK
# ======================================

print("\n======================================")
print("DUPLICATE CUSTOMER VALIDATION")
print("======================================")

duplicate_count = (
    df.groupBy("customerID")
    .count()
    .filter(col("count") > 1)
    .count()
)

print("Duplicate customer IDs:", duplicate_count)

if duplicate_count == 0:
    print("No duplicate customer IDs found.")


# ======================================
# 4. SENIOR CITIZEN VALIDATION
# ======================================

print("\n======================================")
print("SENIOR CITIZEN VALIDATION")
print("======================================")

invalid_senior = df.filter(
    ~col("SeniorCitizen").isin([0, 1])
).count()

print("Invalid SeniorCitizen values:", invalid_senior)


# ======================================
# 5. TENURE VALIDATION
# ======================================

print("\n======================================")
print("TENURE VALIDATION")
print("======================================")

invalid_tenure = df.filter(
    (col("tenure") < 0) |
    (col("tenure") > 100)
).count()

print("Invalid tenure values:", invalid_tenure)


# ======================================
# 6. MONTHLY CHARGES VALIDATION
# ======================================

print("\n======================================")
print("MONTHLY CHARGES VALIDATION")
print("======================================")

invalid_monthly = df.filter(
    col("MonthlyCharges") < 0
).count()

print("Invalid MonthlyCharges values:", invalid_monthly)


# ======================================
# 7. TOTAL CHARGES VALIDATION
# ======================================

print("\n======================================")
print("TOTAL CHARGES VALIDATION")
print("======================================")

invalid_total = df.filter(
    col("TotalCharges") < 0
).count()

print("Invalid TotalCharges values:", invalid_total)


# ======================================
# 8. CHURN VALIDATION
# ======================================

print("\n======================================")
print("CHURN VALIDATION")
print("======================================")

invalid_churn = df.filter(
    ~col("Churn").isin(["Yes", "No"])
).count()

print("Invalid Churn values:", invalid_churn)


# ======================================
# FINAL VALIDATION RESULT
# ======================================

print("\n======================================")
print("VALIDATION SUMMARY")
print("======================================")

validation_failed = (
    len(missing_columns) > 0
    or duplicate_count > 0
    or invalid_senior > 0
    or invalid_tenure > 0
    or invalid_monthly > 0
    or invalid_total > 0
    or invalid_churn > 0
)

if validation_failed:
    print("❌ DATA VALIDATION FAILED")
    raise ValueError(
        "Dataset failed one or more validation checks."
    )
else:
    print("✅ DATA VALIDATION PASSED")
    print("Dataset is ready for feature engineering.")


spark.stop()