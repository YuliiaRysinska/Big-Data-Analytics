from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# ---------------------------------------
# 1. Spark Session
# ---------------------------------------
spark = SparkSession.builder \
    .appName("HorseRacingML") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------
# 2. Load Data
# ---------------------------------------
df = spark.read.csv(
    "horse_racing_results.csv",
    header=True,
    inferSchema=True
)

# ---------------------------------------
# 3. Numeric Feature Cleaning
# ---------------------------------------

# Distance: "2m5f" → miles
df = df.withColumn(
    "dist_num",
    when(col("dist").contains("m"),
         split(col("dist"), "m")[0].cast("double") +
         when(split(col("dist"), "m")[1].contains("f"),
              split(split(col("dist"), "m")[1], "f")[0].cast("double") / 8
         ).otherwise(0)
    )
)

# Weight: "11-9" → stones + pounds
df = df.withColumn(
    "wgt_num",
    when(col("wgt").contains("-"),
         split(col("wgt"), "-")[0].cast("double") * 14 +
         split(col("wgt"), "-")[1].cast("double")
    )
)

# Finishing position
df = df.withColumn("pos_num", col("pos").cast("double"))

# Margin of victory
df = df.withColumn("btn_num", col("btn").cast("double"))

# ---------------------------------------
# 4. Convert Time to Seconds (NO UDF)
# ---------------------------------------
df = df.withColumn(
    "time_sec",
    when(
        col("time").contains(":"),
        split(col("time"), ":")[0].cast("double") * 60 +
        split(col("time"), ":")[1].cast("double")
    )
)

# ---------------------------------------
# 5. Drop Missing Core Values
# ---------------------------------------
df = df.dropna(subset=[
    "dist_num", "wgt_num", "age",
    "time_sec", "btn_num", "pos_num",
    "going", "sex"
])

# ---------------------------------------
# 6. Train / Test Split (SAFE)
# ---------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# ---------------------------------------
# 7. Feature Engineering
# ---------------------------------------
numeric_features = ["dist_num", "wgt_num", "age"]
categorical_features = ["going", "sex"]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="skip"
    )
    for c in categorical_features
]

assembler = VectorAssembler(
    inputCols=numeric_features + [f"{c}_idx" for c in categorical_features],
    outputCol="features",
    handleInvalid="skip"
)

# ---------------------------------------
# 8. TASK 1: Predict Finishing Time
# ---------------------------------------
lr_time = LinearRegression(
    featuresCol="features",
    labelCol="time_sec"
)

pipeline_time = Pipeline(stages=indexers + [assembler, lr_time])
time_model = pipeline_time.fit(train_df)
time_preds = time_model.transform(test_df)

eval_rmse = RegressionEvaluator(
    labelCol="time_sec",
    predictionCol="prediction",
    metricName="rmse"
)

eval_r2 = RegressionEvaluator(
    labelCol="time_sec",
    predictionCol="prediction",
    metricName="r2"
)

print("Time Prediction RMSE:", eval_rmse.evaluate(time_preds))
print("Time Prediction R2:", eval_r2.evaluate(time_preds))

# ---------------------------------------
# 9. TASK 2: Predict Margin of Victory
# ---------------------------------------
lr_margin = LinearRegression(
    featuresCol="features",
    labelCol="btn_num"
)

pipeline_margin = Pipeline(stages=indexers + [assembler, lr_margin])
margin_model = pipeline_margin.fit(train_df)
margin_preds = margin_model.transform(test_df)

print(
    "Margin RMSE:",
    RegressionEvaluator(
        labelCol="btn_num",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(margin_preds)
)

# ---------------------------------------
# 10. TASK 3: Predict Finishing Position
# ---------------------------------------
dt_pos = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="pos_num",
    maxDepth=6,
    maxBins=64
)

pipeline_pos = Pipeline(stages=indexers + [assembler, dt_pos])
pos_model = pipeline_pos.fit(train_df)
pos_preds = pos_model.transform(test_df)

print(
    "Position RMSE:",
    RegressionEvaluator(
        labelCol="pos_num",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(pos_preds)
)

# ---------------------------------------
# 11. Sample Predictions
# ---------------------------------------
pos_preds.select(
    "horse",
    "pos_num",
    "prediction"
).show(10, truncate=False)
