import os, glob
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (SparkSession.builder.appName("Ingest_Netflix_Prize").getOrCreate())

#  Resolve paths (prefer LOCAL with file://, else HDFS) 
LOCAL_DIR = os.path.abspath("data/raw/netflix_prize")
HDFS_DIR  = "hdfs://localhost:9000/projects/netflix_mood/raw_src/netflix_prize"

def exists_local(relpath):
    return os.path.exists(os.path.join(LOCAL_DIR, relpath))

have_local = (
    os.path.isdir(LOCAL_DIR)
    and (exists_local("movie_titles.csv") or exists_local("movie_titles.txt"))
    and len(glob.glob(os.path.join(LOCAL_DIR, "combined_data_*.txt"))) > 0
)

if have_local:
    SRC = f"file://{LOCAL_DIR}"
    TITLES_CSV  = f"{SRC}/movie_titles.csv"
    TITLES_TXT  = f"{SRC}/movie_titles.txt"
    RATINGS_GLOB = f"{SRC}/combined_data_*.txt"
    print(f"📦 Using LOCAL Netflix Prize data at: {LOCAL_DIR}")
else:
    SRC = HDFS_DIR
    TITLES_CSV  = f"{HDFS_DIR}/movie_titles.csv"
    TITLES_TXT  = f"{HDFS_DIR}/movie_titles.txt"
    RATINGS_GLOB = f"{HDFS_DIR}/combined_data_*.txt"
    print(f"🌐 Using HDFS Netflix Prize data at: {HDFS_DIR}")

# Read titles (CSV preferred; fallback to TXT if present) 
titles = None

def read_titles_csv(path_csv):
    df = (spark.read
            .option("header", False)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .option("encoding", "ISO-8859-1")
            .csv(path_csv))
    return df.select(
        F.col("_c0").cast("int").alias("nf_movie_id"),
        F.col("_c1").cast("int").alias("nf_release_year"),
        F.col("_c2").alias("nf_title_raw")
    )

def read_titles_txt(path_txt):
    txt_df = spark.read.text(path_txt)
    def parse_line(line: str):
        parts = line.split(",", 2)
        if len(parts) < 3:
            return None
        try:
            mid = int(parts[0])
            yr  = int(parts[1]) if parts[1].isdigit() else None
            ttl = parts[2]
            return (mid, yr, ttl)
        except:
            return None
    return (txt_df.rdd
                 .map(lambda r: parse_line(r.value))
                 .filter(lambda x: x is not None)
                 .toDF(["nf_movie_id","nf_release_year","nf_title_raw"]))


try:
    titles = read_titles_csv(TITLES_CSV)
    print(f"✅ Read titles CSV: {TITLES_CSV}")
except Exception as e_csv:
    print(f"ℹ️ CSV titles not available or failed ({e_csv}); trying TXT…")
    try:
        titles = read_titles_txt(TITLES_TXT)
        print(f"✅ Read titles TXT: {TITLES_TXT}")
    except Exception as e_txt:
        raise RuntimeError(
            f"Could not read titles. CSV error: {e_csv} | TXT error: {e_txt}"
        )

titles = titles.withColumn("nf_title", F.trim(F.col("nf_title_raw")))

# Read ratings (combined_data_*.txt) 
print(f"📄 Reading ratings from: {RATINGS_GLOB}")
lines = (spark.read.text(RATINGS_GLOB)
         .repartition(8)
         .withColumnRenamed("value", "raw"))

is_movie = F.col("raw").rlike(r"^\d+:$")
df = (lines
      .withColumn("movie_header", is_movie)
      .withColumn("nf_movie_id",
                  F.when(is_movie, F.regexp_extract("raw", r"^(\d+):$", 1).cast("int"))))

w = Window.orderBy(F.monotonically_increasing_id())
df = df.withColumn("nf_movie_id_ff", F.last("nf_movie_id", ignorenulls=True).over(w))

ratings = (df.filter(~F.col("movie_header"))
             .withColumn("user_id", F.split("raw", ",").getItem(0).cast("int"))
             .withColumn("rating",  F.split("raw", ",").getItem(1).cast("int"))
             .withColumn("date",    F.to_date(F.split("raw", ",").getItem(2), "yyyy-MM-dd"))
             .drop("raw","movie_header","nf_movie_id")
             .withColumnRenamed("nf_movie_id_ff","nf_movie_id"))

# Write to HDFS curated/raw 
OUT_TITLES  = "/projects/netflix_mood/raw/netflix_prize_titles"
OUT_RATINGS = "/projects/netflix_mood/raw/netflix_prize_ratings"

titles.write.mode("overwrite").parquet(OUT_TITLES)
ratings.write.mode("overwrite").parquet(OUT_RATINGS)

print(f"✅ Wrote {OUT_TITLES} and {OUT_RATINGS}")
spark.stop()
