from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, trim


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("CustomerChurnFeatureEngineering") \
    .getOrCreate()

print("\n======================================")
print("CUSTOMER CHURN FEATURE ENGINEERING")
print("======================================")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

project_path = "/content/drive/MyDrive/Customer Churn Prediction Pipeline"

cleaned_path = f"{project_path}/data/processed/cleaned_churn"

engineered_path = f"{project_path}/data/processed/engineered_churn"


# ============================================================
# 3. LOAD CLEANED DATA
# ============================================================

df = spark.read.csv(
    cleaned_path,
    header=True,
    inferSchema=True
)

print("\nCleaned dataset loaded successfully!")

print("Rows:", df.count())
print("Columns:", len(df.columns))


# ============================================================
# 4. HANDLE TOTAL CHARGES
# ============================================================

# Remove spaces
df = df.withColumn(
    "TotalCharges",
    trim(col("TotalCharges"))
)

# Convert empty strings to NULL
df = df.withColumn(
    "TotalCharges",
    when(
        col("TotalCharges") == "",
        None
    ).otherwise(col("TotalCharges"))
)

# Convert to double
df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast("double")
)

# Replace missing TotalCharges with 0
df = df.withColumn(
    "TotalCharges",
    when(
        col("TotalCharges").isNull(),
        0.0
    ).otherwise(col("TotalCharges"))
)


# ============================================================
# 5. CREATE ENGINEERED FEATURES
# ============================================================

# ------------------------------------------------------------
# Average Monthly Charge
# ------------------------------------------------------------

df = df.withColumn(
    "AverageMonthlyCharge",
    when(
        col("tenure") > 0,
        col("TotalCharges") / col("tenure")
    ).otherwise(0.0)
)


# ------------------------------------------------------------
# Is New Customer
# ------------------------------------------------------------

df = df.withColumn(
    "IsNewCustomer",
    when(
        col("tenure") <= 6,
        1
    ).otherwise(0)
)


# ------------------------------------------------------------
# Is Month-to-Month Customer
# ------------------------------------------------------------

df = df.withColumn(
    "IsMonthToMonth",
    when(
        col("Contract") == "Month-to-month",
        1
    ).otherwise(0)
)


# ------------------------------------------------------------
# Has Technical Support
# ------------------------------------------------------------

df = df.withColumn(
    "HasTechSupport",
    when(
        col("TechSupport") == "Yes",
        1
    ).otherwise(0)
)


# ------------------------------------------------------------
# Has Online Security
# ------------------------------------------------------------

df = df.withColumn(
    "HasOnlineSecurity",
    when(
        col("OnlineSecurity") == "Yes",
        1
    ).otherwise(0)
)


# ------------------------------------------------------------
# Has Streaming
# ------------------------------------------------------------

df = df.withColumn(
    "HasStreaming",
    when(
        (col("StreamingTV") == "Yes") |
        (col("StreamingMovies") == "Yes"),
        1
    ).otherwise(0)
)


print("\nFeature engineering completed successfully!")


# ============================================================
# 6. DISPLAY ENGINEERED FEATURES
# ============================================================

print("\nSample engineered data:")

df.select(
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlyCharge",
    "IsNewCustomer",
    "IsMonthToMonth",
    "HasTechSupport",
    "HasOnlineSecurity",
    "HasStreaming",
    "Churn"
).show(
    10,
    truncate=False
)


# ============================================================
# 7. SAVE ENGINEERED DATA
# ============================================================

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(engineered_path)


print("\n======================================")
print("ENGINEERED DATA SAVED")
print("======================================")

print("Output path:")
print(engineered_path)


# ============================================================
# 8. FINISH
# ============================================================

spark.stop()

print("\nFeature engineering completed!")