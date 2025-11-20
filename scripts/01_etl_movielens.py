from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("ETL_MovieLens")
    .getOrCreate()
)

# Read MovieLens CSVs from HDFS
ratings = (
    spark.read
    .option("header", True)
    .csv("/projects/netflix_mood/raw/movielens/ratings.csv")
)

movies = (
    spark.read
    .option("header", True)
    .csv("/projects/netflix_mood/raw/movielens/movies.csv")
)

tags_raw = (
    spark.read
    .option("header", True)
    .csv("/projects/netflix_mood/raw/movielens/tags.csv")
)


ratings = ratings.select(
    F.col("userId").cast("int").alias("userId"),
    F.col("movieId").cast("int").alias("movieId"),
    F.col("rating").cast("double").alias("rating"),
    F.col("timestamp").cast("long").alias("timestamp")
)

# Movies: cast movieId + splitting genres
movies = (
    movies
    .select(
        F.col("movieId").cast("int").alias("movieId"),
        F.col("title").alias("title"),
        F.col("genres").alias("genres")
    )
    .withColumn("genres_array", F.split(F.col("genres"), "\\|"))
)

tags = (
    tags_raw
    .withColumn("userId", F.col("userId").cast("int"))
    .withColumn("movieId", F.col("movieId").cast("int"))
    .withColumn("tag", F.col("tag"))
    .withColumn(
        "timestamp_clean",
        F.regexp_replace(F.col("timestamp"), "[^0-9]", "")
    )
    .withColumn(
        "timestamp_clean",
        F.when(F.col("timestamp_clean") == "", None).otherwise(F.col("timestamp_clean"))
    )

    .withColumn("timestamp", F.col("timestamp_clean").cast("long"))
    .drop("timestamp_clean")
    .select("userId", "movieId", "tag", "timestamp")
)

# curated parquet tables
ratings.write.mode("overwrite").parquet("/projects/netflix_mood/curated/movielens_ratings")
movies.write.mode("overwrite").parquet("/projects/netflix_mood/curated/movielens_movies")
tags.write.mode("overwrite").parquet("/projects/netflix_mood/curated/movielens_tags")

print("✅ ETL completed successfully.")
spark.stop()
