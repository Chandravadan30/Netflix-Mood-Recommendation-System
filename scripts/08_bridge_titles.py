from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.appName("Bridge_Titles").getOrCreate()


ML_MOVIES_PATH = "/projects/netflix_mood/curated/movielens_movies"          # columns: movieId, title, genres, ...
NF_TITLES_PATH = "/projects/netflix_mood/raw/netflix_prize_titles"          # columns: nf_movie_id, nf_release_year, nf_title, ...

def canon_expr(col):
    """
    Canonicalize a title:
      1) remove trailing year in parentheses
      2) drop leading articles (the|a|an)
      3) keep [a-z0-9] only
      4) lowercase & trim
    """
    no_year = F.regexp_replace(col, r"\s*\(\d{4}\)\s*$", "")
    no_article = F.regexp_replace(F.lower(no_year), r"^\s*(the|a|an)\s+", "")
    compact = F.regexp_replace(no_article, r"[^a-z0-9]+", "")
    return F.trim(compact)

#  Read MovieLens 
ml = spark.read.parquet(ML_MOVIES_PATH)

# Extract year from "Toy Story (1995)" -> 1995
ml = ml.withColumn("ml_year", F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).cast("int")) \
       .withColumn("ml_canon", canon_expr(F.col("title"))) \
       .select("movieId", "title", "ml_year", "ml_canon")

#  Read Netflix Prize Titles 
nf = spark.read.parquet(NF_TITLES_PATH) \
       .withColumn("nf_canon", canon_expr(F.col("nf_title"))) \
       .select("nf_movie_id", "nf_title", "nf_release_year", "nf_canon")

#  Join strategy 
#  Strict join on (canon, year) when both years exist
strict_join = nf.join(
    ml,
    (nf.nf_canon == ml.ml_canon) & (nf.nf_release_year.isNotNull()) & (ml.ml_year.isNotNull()) & (nf.nf_release_year == ml.ml_year),
    "inner"
).select(
    "nf_movie_id", "nf_title", "nf_release_year",
    "movieId", "title", "ml_year", "nf_canon"
)

# Fallback join on canon only for records not matched above
strict_keys = strict_join.select("nf_movie_id").dropDuplicates()
nf_unmatched = nf.join(strict_keys, "nf_movie_id", "left_anti")

fallback_join = nf_unmatched.join(
    ml,
    nf_unmatched.nf_canon == ml.ml_canon,
    "inner"
).select(
    "nf_movie_id", "nf_title", "nf_release_year",
    "movieId", "title", "ml_year", "nf_canon"
)

# Union and de-dup (prefer strict matches where duplicates exist)
bridge_all = strict_join.unionByName(fallback_join) \
                        .withColumn("pref", F.when(F.col("nf_release_year") == F.col("ml_year"), F.lit(1)).otherwise(F.lit(0)))

w = Window.partitionBy("nf_movie_id", "movieId").orderBy(F.desc("pref"))
bridge = bridge_all.withColumn("rn", F.row_number().over(w)) \
                   .filter(F.col("rn") == 1) \
                   .drop("pref", "rn")

#  Write out 
OUT_PATH = "/projects/netflix_mood/curated/bridge_netflixprize_movielens"
bridge.write.mode("overwrite").parquet(OUT_PATH)

# Small log
print("✅ Bridge written to", OUT_PATH)
print("Counts:",
      "NF titles =", nf.count(),
      "ML titles =", ml.count(),
      "Bridged pairs =", bridge.count())

spark.stop()
