from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.appName("GenerateRecs").getOrCreate()
clusters = spark.read.parquet("/projects/netflix_mood/curated/clustered_movies")

w = Window.partitionBy("cluster").orderBy(F.desc("compound"), F.desc("release_year"))
top_by_cluster = (clusters
                  .withColumn("rank_in_cluster", F.row_number().over(w))
                  .filter(F.col("rank_in_cluster") <= 50))

# Save per cluster and an overall per-mood top list
top_by_cluster.write.mode("overwrite").parquet("/projects/netflix_mood/recommendations/mood_top50")

top_by_mood = (clusters
               .withColumn("rank_in_mood",
                           F.row_number().over(Window.partitionBy("mood")
                                              .orderBy(F.desc("compound"), F.desc("release_year"))))
               .filter(F.col("rank_in_mood") <= 100))
top_by_mood.write.mode("overwrite").parquet("/projects/netflix_mood/recommendations/mood_top100")

print("✅ Recommendations saved under /projects/netflix_mood/recommendations/")
spark.stop()
