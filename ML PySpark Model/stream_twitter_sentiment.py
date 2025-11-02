# stream_twitter_sentiment.py

import tweepy
import json
import os
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# -------------------------------
# 1. Twitter API setup
# -------------------------------
bearer_token = "YOUR_TWITTER_BEARER_TOKEN"  # replace with your token
query = "Borderlands -is:retweet lang:en"  # topic filter

# -------------------------------
# 2. PySpark setup
# -------------------------------
spark = SparkSession.builder \
    .appName("LiveTweetSentiment") \
    .getOrCreate()

# Load your trained Logistic Regression pipeline
model_path = "/home/aleema/Downloads/Real-Time-Twitter-Sentiment-Analysis/ML PySpark Model/logistic_regression_model.pkl"
pipeline_model = PipelineModel.load(model_path)

# UDF to map numeric prediction to labels
def map_label(pred):
    return "Positive" if pred == 1.0 else "Negative" if pred == 0.0 else "Neutral"

label_udf = udf(map_label, StringType())

# Output CSV file
output_file = "live_tweets_labeled.csv"

# Create file with header if not exists
if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TweetID\tEntity\tSentiment\tTweetContent\n")

# Tweet ID counter (to avoid duplicates)
tweet_id_counter = 1000000  # start after your existing dataset

# -------------------------------
# 3. Twitter streaming using Tweepy
# -------------------------------
class MyStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        global tweet_id_counter

        # Create Spark DataFrame
        df = spark.createDataFrame([Row(
            TweetID=tweet_id_counter,
            Entity="Borderlands",
            TweetContent=tweet.text
        )])

        # Predict sentiment
        predictions = pipeline_model.transform(df)
        results = predictions.withColumn("Sentiment", label_udf("prediction")).drop("prediction")
        
        # Convert to Pandas for writing to CSV
        pdf = results.toPandas()
        pdf.to_csv(output_file, mode="a", header=False, index=False, sep="\t", encoding="utf-8")

        print(f"Saved TweetID {tweet_id_counter} -> {pdf['Sentiment'][0]}")

        tweet_id_counter += 1

    def on_error(self, status):
        print("Error:", status)
        return True  # keep stream alive

# Initialize stream
stream = MyStream(bearer_token)
stream.add_rules(tweepy.StreamRule(query))

# Start streaming (this runs indefinitely)
stream.filter(tweet_fields=["text"])
