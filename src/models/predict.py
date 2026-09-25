from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import when, col
from pyspark.ml.functions import vector_to_array

import sys
import os


# ======================================
# SPARK SESSION
# ======================================

spark = SparkSession.builder \
    .appName("CustomerChurnBatchPrediction") \
    .getOrCreate()


print("\n======================================")
print("CUSTOMER CHURN BATCH PREDICTION")
print("======================================")


# ======================================
# PROJECT PATHS
# ======================================

project_path = "/content/drive/MyDrive/Customer Churn Prediction Pipeline"

model_path = f"{project_path}/models/churn_pipeline"


# ======================================
# INPUT FILE
# ======================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python predict.py <input_csv>")
    spark.stop()
    sys.exit(1)

input_path = sys.argv[1]

print("\nInput file:")
print(input_path)


# ======================================
# CHECK INPUT FILE
# ======================================

if not os.path.exists(input_path):
    print("\nERROR: Input file does not exist.")
    spark.stop()
    sys.exit(1)


# ======================================
# LOAD TRAINED PIPELINE
# ======================================

print("\nLoading trained pipeline...")

pipeline_model = PipelineModel.load(model_path)

print("Pipeline loaded successfully!")


# ======================================
# LOAD NEW CUSTOMER DATA
# ======================================

print("\nLoading new customer data...")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True,
    sep=","
)


print("Input data loaded successfully!")
print("Rows:", df.count())
print("Columns:", len(df.columns))




print("\nInput schema:")
df.printSchema()
# ======================================
# FEATURE ENGINEERING
# ======================================

print("\nApplying feature engineering...")

df = df.withColumn(
    "AverageMonthlyCharge",
    when(
        col("tenure") > 0,
        col("TotalCharges") / col("tenure")
    ).otherwise(0.0)
)

df = df.withColumn(
    "IsNewCustomer",
    when(
        col("tenure") <= 6,
        1
    ).otherwise(0)
)

df = df.withColumn(
    "IsMonthToMonth",
    when(
        col("Contract") == "Month-to-month",
        1
    ).otherwise(0)
)

df = df.withColumn(
    "HasTechSupport",
    when(
        col("TechSupport") == "Yes",
        1
    ).otherwise(0)
)

df = df.withColumn(
    "HasOnlineSecurity",
    when(
        col("OnlineSecurity") == "Yes",
        1
    ).otherwise(0)
)

df = df.withColumn(
    "HasStreaming",
    when(
        (col("StreamingTV") == "Yes") |
        (col("StreamingMovies") == "Yes"),
        1
    ).otherwise(0)
)

print("Feature engineering completed!")


# ======================================
# GENERATE PREDICTIONS
# ======================================

print("\nGenerating predictions...")

predictions = pipeline_model.transform(df)


# ======================================
# CREATE FINAL OUTPUT
# ======================================

result = predictions.select(
    "customerID",
    "prediction",
    "probability"
)

result = result.withColumn(
    "churn_prediction",
    when(col("prediction") == 1.0, "Yes").otherwise("No")
)
result = result.withColumn(
    "churn_probability",
    vector_to_array(col("probability"))[1]
)

result = result.select(
    "customerID",
    "churn_prediction",
    "churn_probability"
)


# ======================================
# DISPLAY RESULTS
# ======================================

print("\n======================================")
print("PREDICTION RESULTS")
print("======================================")

result.show(
    20,
    truncate=False
)


# ======================================
# SAVE RESULTS
# ======================================

output_path = f"{project_path}/predictions"

result.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


print("\n======================================")
print("PREDICTIONS SAVED")
print("======================================")

print("Output path:")
print(output_path)


spark.stop()

print("\nPrediction completed successfully!")