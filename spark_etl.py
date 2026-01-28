from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, trim, col
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import os

def create_spark_session():
    """Initialize Spark session"""
    print("Creating Spark session...")
    spark = SparkSession.builder \
        .appName("SupportTicketClassifier") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    print("✓ Spark session created")
    return spark

def load_data_spark(spark, filepath='data/tickets.csv'):
    """Load data using Spark"""
    print(f"Loading data from {filepath} with PySpark...")
    
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    count = df.count()
    print(f"✓ Loaded {count} tickets with PySpark")
    return df

def validate_data_spark(df):
    """Validate and clean data using Spark"""
    print("Validating data quality with PySpark...")
    initial_count = df.count()
    
    # Remove nulls
    df = df.dropna(subset=['text', 'category'])
    
    # Remove duplicates
    df = df.dropDuplicates(['text'])
    
    removed = initial_count - df.count()
    print(f"✓ Removed {removed} invalid/duplicate records")
    return df

def preprocess_text_spark(df):
    """Preprocess text using Spark"""
    print("Preprocessing text with PySpark...")
    
    # Lowercase and trim
    df = df.withColumn('text', lower(trim(col('text'))))
    
    print("✓ Text preprocessing complete")
    return df

def feature_engineering_spark(df):
    """Extract features using Spark ML"""
    print("Extracting features with Spark ML (TF-IDF)...")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=100)
    
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])

    model = pipeline.fit(df)
    feature_df = model.transform(df)
    
    print("✓ Feature extraction complete with Spark ML")
    return feature_df, model

def run_spark_etl(input_path='data/tickets.csv'):
    """Run complete Spark ETL pipeline"""
    print("\n" + "="*60)
    print("RUNNING PYSPARK ETL PIPELINE")
    print("="*60 + "\n")
    
    # spare initializing
    spark = create_spark_session()
    df = load_data_spark(spark, input_path)
    df = validate_data_spark(df)
    df = preprocess_text_spark(df)
    df_features, feature_model = feature_engineering_spark(df)
    
    print("\n" + "="*60)
    print("✓ PYSPARK ETL PIPELINE COMPLETE")
    print("="*60)
    print(f"\nFinal dataset: {df_features.count()} tickets")
    print(f"Feature columns: {df_features.columns}")
    
    return df_features, spark

if __name__ == "__main__":
    df, spark = run_spark_etl()
    # sample data
    print("\nSample processed data:")
    df.select("text", "category", "features").show(5, truncate=50)
    spark.stop()
    print("\n✓ Spark session stopped")
