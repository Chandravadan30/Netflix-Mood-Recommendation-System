from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

spark = SparkSession.builder.appName("BuildFeatures_clean").getOrCreate()
sent = spark.read.parquet("/projects/netflix_mood/features/nf_sentiment")

sent = sent.filter(F.col("compound").isNotNull() & F.col("release_year").isNotNull())

#2) Canonicalize category into a small set
def canon_cat(c):
    c = (c or "General").lower()
    # take first token before comma
    c = c.split(",")[0].strip()
    # simple mapping
    if "drama" in c: return "Drama"
    if "comedy" in c: return "Comedy"
    if "romance" in c: return "Romance"
    if "action" in c: return "Action"
    if "thriller" in c or "crime" in c: return "Thriller"
    if "document" in c: return "Documentary"
    if "animation" in c or "anime" in c: return "Animation"
    if "family" in c or "children" in c: return "Family"
    if "horror" in c: return "Horror"
    if "scifi" in c or "sci-fi" in c or "science" in c or "fantasy" in c: return "SciFiFantasy"
    return "General"

canon_cat_udf = F.udf(canon_cat)
sent = sent.withColumn("main_cat", canon_cat_udf(F.col("categories")))

# Year as a smooth cyclic proxy: center + sin/cos 
year = F.col("release_year").cast("double")
sent = sent.withColumn("year_centered", year - F.lit(2000.0))
sent = sent.withColumn("year_sin", F.sin(F.col("year_centered")/10.0))
sent = sent.withColumn("year_cos", F.cos(F.col("year_centered")/10.0))

#Small categorical encoding 
idx = StringIndexer(inputCol="main_cat", outputCol="main_cat_idx", handleInvalid="keep").fit(sent)
with_idx = idx.transform(sent)

ohe = OneHotEncoder(inputCols=["main_cat_idx"], outputCols=["main_cat_ohe"], dropLast=True)
with_ohe = ohe.fit(with_idx).transform(with_idx)

#Assemble & scale 
assembler = VectorAssembler(
    inputCols=["compound", "year_sin", "year_cos", "main_cat_ohe"],
    outputCol="features_raw", handleInvalid="keep"
)
assembled = assembler.transform(with_ohe)

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaled = scaler.fit(assembled).transform(assembled)

final = scaled.select(
    "nf_title","nf_description","categories","release_year","mood","compound",
    "main_cat","features"
)

final.write.mode("overwrite").parquet("/projects/netflix_mood/features/nf_features")
print("✅ Clean features written to /projects/netflix_mood/features/nf_features")
spark.stop()
print("✅ Success")