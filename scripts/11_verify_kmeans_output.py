from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Verify_KMeans_Output").getOrCreate()
pred = spark.read.parquet("/projects/netflix_mood/models/kmeans_predictions")

print("Columns:", pred.columns)
pred.select("nf_title", "cluster").show(10, truncate=False)
pred.groupBy("cluster").count().orderBy("cluster").show()

spark.stop()
print("✅ KMeans predictions verified")