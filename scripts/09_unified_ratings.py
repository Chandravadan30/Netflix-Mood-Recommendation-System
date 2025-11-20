from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("Unified_Ratings").getOrCreate()

# Bridge between Netflix IDs and MovieLens IDs
bridge = (
    spark.read.parquet("/projects/netflix_mood/curated/bridge_netflixprize_movielens")
         .select("nf_movie_id", "movieId")
)

# MovieLens ratings
ml_ratings = (
    spark.read.parquet("/projects/netflix_mood/curated/movielens_ratings")
         .select("userId", "movieId", "rating", "timestamp")
)

# Netflix Prize ratings – use existing nf_movie_id column, but drop NULLs
nf_ratings = (
    spark.read.parquet("/projects/netflix_mood/raw/netflix_prize_ratings")
         .select("nf_movie_id", "user_id", "rating", "date")
         .where(F.col("nf_movie_id").isNotNull())      # 🔑 avoid NULL keys
         .withColumnRenamed("user_id", "userId")
         .withColumn("source", F.lit("netflix_prize"))
)

# Attach ML with bridge and tag source
ml_joined = (
    ml_ratings.join(bridge, "movieId", "inner")
              .select("userId", "movieId", "nf_movie_id", "rating", "timestamp")
              .withColumn("source", F.lit("movielens"))
)

# Attach NF with bridge (only rows with matching nf_movie_id)
nf_joined = (
    nf_ratings.join(bridge, "nf_movie_id", "inner")
              .select("userId", "movieId", "nf_movie_id", "rating", "date", "source")
)

# Normalize timestamps
ml_final = (
    ml_joined.withColumn("event_ts", (F.col("timestamp") * 1000).cast("timestamp"))
             .drop("timestamp")
)

nf_final = (
    nf_joined.withColumn("event_ts", F.col("date").cast("timestamp"))
             .drop("date")
)

unified = ml_final.unionByName(nf_final, allowMissingColumns=True)

out = "/projects/netflix_mood/curated/unified_ratings"
unified.write.mode("overwrite").parquet(out)

print("✅ Unified ratings ->", out)
print("ML rows:", ml_final.count())
print("NF rows:", nf_final.count())
print("Total rows:", unified.count())

spark.stop()
print("✅ Unified ratings verified")