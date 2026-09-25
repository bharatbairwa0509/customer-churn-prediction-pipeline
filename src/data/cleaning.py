from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, trim, sum

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("CustomerChurnDataCleaning") \
    .getOrCreate()

# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

project_path = "/content/drive/MyDrive/Customer Churn Prediction Pipeline"

raw_file = f"{project_path}/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

output_path = f"{project_path}/data/processed/cleaned_churn"

# --------------------------------------------------
# 3. Read raw CSV
# --------------------------------------------------

df = spark.read.csv(
    raw_file,
    header=True,
    inferSchema=True
)

print("Original rows:", df.count())
print("Original columns:", len(df.columns))

# --------------------------------------------------
# 4. Remove duplicate rows
# --------------------------------------------------

before = df.count()

df = df.dropDuplicates()

after = df.count()

print("Rows before duplicates:", before)
print("Rows after duplicates:", after)
print("Duplicates removed:", before - after)

# --------------------------------------------------
# 5. Clean TotalCharges
# --------------------------------------------------

df = df.withColumn(
    "TotalCharges",
    trim(col("TotalCharges"))
)

df = df.withColumn(
    "TotalCharges",
    when(col("TotalCharges") == "", None)
    .otherwise(col("TotalCharges"))
)

df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast("double")
)

# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

print("\nMissing values:")

missing_values = df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

missing_values.show()

# --------------------------------------------------
# 7. Check schema
# --------------------------------------------------

print("\nSchema after cleaning:")

df.printSchema()

# --------------------------------------------------
# 8. Check invalid numeric values
# --------------------------------------------------

print("\nInvalid SeniorCitizen values:")

df.filter(
    ~col("SeniorCitizen").isin([0, 1])
).show()

print("\nInvalid tenure values:")

df.filter(
    (col("tenure") < 0) | (col("tenure") > 72)
).show()

print("\nInvalid MonthlyCharges values:")

df.filter(
    col("MonthlyCharges") < 0.0
).show()

print("\nInvalid TotalCharges values:")

df.filter(
    col("TotalCharges") < 0.0
).show()

# --------------------------------------------------
# 9. Save cleaned dataset
# --------------------------------------------------

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nCleaned dataset saved successfully!")
print("Output path:", output_path)