from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("Prep_Text").getOrCreate()

nf_csv = "/projects/netflix_mood/raw/metadata/netflix_titles.csv"

use_netflix = True
try:
    spark.read.option("header", True).csv(nf_csv).limit(1).count()
except Exception:
    use_netflix = False

if use_netflix:
    titles = (spark.read.option("header", True).csv(nf_csv)
              .filter(F.lower(F.col("type")) == "movie"))
    movies_meta = (titles
        .select(
            F.col("title").alias("nf_title"),
            F.col("description").alias("nf_description"),
            F.col("listed_in").alias("categories"),
            F.col("release_year").cast("int").alias("release_year"))
        .dropna(subset=["nf_title"]))
else:
    ml_movies = spark.read.parquet("/projects/netflix_mood/curated/movielens_movies")
    # Extract year from "(YYYY)" in title using a correctly-escaped regex
    movies_meta = (ml_movies
        .select(
            F.col("title").alias("nf_title"),
            F.when((F.col("genres").isNull()) | (F.col("genres") == "(no genres listed)"),
                   F.lit("General")).otherwise(F.col("genres")).alias("categories"),
            F.lit(None).cast("string").alias("nf_description"),
            F.regexp_extract(F.col("title"), r".*\((\d{4})\)", 1).cast("int").alias("release_year"))
        .withColumn(
            "nf_description",
            F.coalesce(
                F.col("nf_description"),
                F.concat_ws(" ", F.lit("A movie tagged as"), F.col("categories"), F.lit("titled"), F.col("nf_title"))
            ))
    )

movies_meta.write.mode("overwrite").parquet("/projects/netflix_mood/curated/nf_movies_meta")
print("✅ Prepared text metadata at /projects/netflix_mood/curated/nf_movies_meta (Netflix or fallback)")
spark.stop()
