from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("KMeansMood").getOrCreate()
data  = spark.read.parquet("/projects/netflix_mood/features/nf_features")

best_k, best_s, best_model = None, -1, None
for k in [3,4,5,6,7,8]:
    km = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    m  = km.fit(data)
    s  = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                             metricName="silhouette").evaluate(m.transform(data))
    if s > best_s:
        best_k, best_s, best_model = k, s, m

pred = best_model.transform(data)
print(f"✅ Best K={best_k} silhouette={best_s:.4f}")

pred.select("nf_title","mood","compound","release_year","categories","cluster") \
    .write.mode("overwrite").parquet("/projects/netflix_mood/curated/clustered_movies")

best_model.write().overwrite().save("/projects/netflix_mood/models/kmeans_model")
spark.stop()

