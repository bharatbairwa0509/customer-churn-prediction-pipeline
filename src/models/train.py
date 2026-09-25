from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import json

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import shutil


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("CustomerChurnTrainingPipeline") \
    .getOrCreate()

print("\n======================================")
print("CUSTOMER CHURN MODEL TRAINING")
print("======================================")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

project_path = "/content/drive/MyDrive/Customer Churn Prediction Pipeline"

engineered_path = f"{project_path}/data/processed/engineered_churn"

model_path = f"{project_path}/models/churn_pipeline"


# ============================================================
# 3. LOAD ENGINEERED DATA
# ============================================================

df = spark.read.csv(
    engineered_path,
    header=True,
    inferSchema=True
)

print("\nEngineered dataset loaded successfully!")

print("Rows:", df.count())
print("Columns:", len(df.columns))


# ============================================================
# 4. CREATE TARGET LABEL
# ============================================================

# Churn = No  -> 0
# Churn = Yes -> 1

df = df.withColumn(
    "label",
    when(
        col("Churn") == "Yes",
        1.0
    ).otherwise(0.0)
)


print("\nTarget label created successfully!")


# ============================================================
# 5. DEFINE NUMERIC FEATURES
# ============================================================

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlyCharge",
    "IsNewCustomer",
    "IsMonthToMonth",
    "HasTechSupport",
    "HasOnlineSecurity",
    "HasStreaming"
]


# ============================================================
# 6. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
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
    "PaymentMethod"
]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)


print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print("Training rows:", train_df.count())
print("Testing rows :", test_df.count())


# ============================================================
# 8. STRING INDEXERS
# ============================================================

indexers = []

for column_name in categorical_cols:

    indexer = StringIndexer(
        inputCol=column_name,
        outputCol=column_name + "_index",
        handleInvalid="keep"
    )

    indexers.append(indexer)


# ============================================================
# 9. ONE-HOT ENCODER
# ============================================================

indexed_cols = [
    column_name + "_index"
    for column_name in categorical_cols
]

encoded_cols = [
    column_name + "_encoded"
    for column_name in categorical_cols
]


encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=encoded_cols
)


# ============================================================
# 10. VECTOR ASSEMBLER
# ============================================================

assembler_inputs = numeric_cols + encoded_cols

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="error"
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=8,
    seed=42
)


# ============================================================
# 12. CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        rf
    ]
)


print("\nComplete ML pipeline created!")


# ============================================================
# 13. TRAIN COMPLETE PIPELINE
# ============================================================

print("\nTraining pipeline...")

pipeline_model = pipeline.fit(train_df)

print("Pipeline training completed successfully!")


# ============================================================
# 14. MAKE TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

predictions = pipeline_model.transform(test_df)


# ============================================================
# 15. SHOW PREDICTIONS
# ============================================================

print("\nSample predictions:")

predictions.select(
    "label",
    "prediction",
    "probability"
).show(
    10,
    truncate=False
)


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)


accuracy = accuracy_evaluator.evaluate(predictions)

f1 = f1_evaluator.evaluate(predictions)

precision = precision_evaluator.evaluate(predictions)

recall = recall_evaluator.evaluate(predictions)


# ============================================================
# 17. DISPLAY MODEL METRICS
# ============================================================

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")


confusion_matrix = predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
)


confusion_matrix.show()


# ============================================================
# 19. CHURN-SPECIFIC METRICS
# ============================================================

true_positives = predictions.filter(
    (col("label") == 1) &
    (col("prediction") == 1)
).count()


false_positives = predictions.filter(
    (col("label") == 0) &
    (col("prediction") == 1)
).count()


false_negatives = predictions.filter(
    (col("label") == 1) &
    (col("prediction") == 0)
).count()


if true_positives + false_positives > 0:

    churn_precision = (
        true_positives /
        (true_positives + false_positives)
    )

else:

    churn_precision = 0.0


if true_positives + false_negatives > 0:

    churn_recall = (
        true_positives /
        (true_positives + false_negatives)
    )

else:

    churn_recall = 0.0



# ======================================
# SAVE MODEL METRICS
# ======================================

reports_path = f"{project_path}/reports"

os.makedirs(reports_path, exist_ok=True)

metrics = {
    "model": "Random Forest",
    "numTrees": 50,
    "maxDepth": 8,
    "seed": 42,
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "churn_precision": float(churn_precision),
    "churn_recall": float(churn_recall),
    "confusion_matrix": {
        "true_negative": int(
            predictions.filter(
                (col("label") == 0) &
                (col("prediction") == 0)
            ).count()
        ),
        "false_positive": int(false_positives),
        "false_negative": int(false_negatives),
        "true_positive": int(true_positives)
    }
}

metrics_file = f"{reports_path}/model_metrics.json"

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n======================================")
print("MODEL METRICS SAVED")
print("======================================")
print(metrics_file)

print("\n======================================")
print("CHURN CLASS METRICS")
print("======================================")

print(f"Churn Precision : {churn_precision:.4f}")
print(f"Churn Recall    : {churn_recall:.4f}")


# ============================================================
# 20. SAVE COMPLETE PIPELINE
# ============================================================

print("\n======================================")
print("SAVING COMPLETE PIPELINE")
print("======================================")


if os.path.exists(model_path):

    shutil.rmtree(model_path)

    print("Existing pipeline removed.")


pipeline_model.write().save(model_path)


print("\nComplete pipeline saved successfully!")

print("Model path:")
print(model_path)


# ============================================================
# 21. FINISH
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("======================================")


spark.stop()