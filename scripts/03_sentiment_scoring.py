from pyspark.sql import SparkSession, functions as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

spark = SparkSession.builder.appName("SentimentScoring").getOrCreate()
meta  = spark.read.parquet("/projects/netflix_mood/curated/nf_movies_meta")

analyzer = SentimentIntensityAnalyzer()
bc = spark.sparkContext.broadcast(analyzer)

@F.udf("double")
def vader_compound(text):
    if text is None: return 0.0
    return float(bc.value.polarity_scores(text)["compound"])

@F.udf("string")
def mood_bucket(c):
    if c <= -0.35:   return "Dark"
    if c < 0.15:     return "Neutral"
    if c < 0.45:     return "Positive"
    return "Inspirational"

scored = (meta
    .withColumn("compound", vader_compound(F.col("nf_description")))
    .withColumn("mood", mood_bucket(F.col("compound")))
)

scored.write.mode("overwrite").parquet("/projects/netflix_mood/features/nf_sentiment")
print("✅ Sentiment + mood ready at /projects/netflix_mood/features/nf_sentiment")
spark.stop()
