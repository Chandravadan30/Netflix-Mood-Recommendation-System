from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("TrainKMeans_clean").getOrCreate()
df = spark.read.parquet("/projects/netflix_mood/features/nf_features")

# Try a small range and choose best by silhouette
cands = [6, 7, 8, 9, 10]
best = None
best_model = None
for k in cands:
    km = KMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction")
    m = km.fit(df)
    pred = m.transform(df)
    sil = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette").evaluate(pred)
    print(f"K={k}, silhouette={sil:.4f}")
    if best is None or sil > best:
        best, best_model = sil, m

pred = best_model.transform(df).withColumnRenamed("prediction", "cluster")
pred.select("nf_title","main_cat","release_year","compound","cluster").write.mode("overwrite").parquet(
    "/projects/netflix_mood/models/kmeans_predictions"
)
print("✅ Saved /projects/netflix_mood/models/kmeans_predictions")
spark.stop()
print("✅ KMeans training complete")