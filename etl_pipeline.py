import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Extract, transform, and load

def load_data(filepath='data/tickets.csv'):
    """Load support ticket data from CSV"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} tickets")
    return df

def validate_data(df):
    """Validate and clean ticket data"""
    print("Validating data quality...")
    initial_count = len(df)
    
    # null values
    df = df.dropna(subset=['text', 'category'])
    
    # duplicates
    df = df.drop_duplicates(subset=['text'])
    
    print(f"✓ Removed {initial_count - len(df)} invalid/duplicate records")
    return df

def preprocess_text(df):
    """Preprocess ticket text"""
    print("Preprocessing text...")
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.strip()
    print("✓ Text preprocessing complete")
    return df

def feature_engineering(df, vectorizer=None, fit=True):
    """Extract features from ticket text using TF-IDF"""
    print("Extracting features")
    
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2
        )
        X = vectorizer.fit_transform(df['text'])
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        print("✓ Fitted new vectorizer")
    else:
        X = vectorizer.transform(df['text'])
        print("✓ Transformed using existing vectorizer")
    
    print(f"✓ Feature matrix shape: {X.shape}")
    return X, vectorizer

def run_etl(input_path='data/tickets.csv', fit_vectorizer=True):
    """Run complete ETL pipeline"""
    print("\n" + "="*50)
    print("RUNNING ETL PIPELINE")
    print("="*50 + "\n")
    
    # models directory
    os.makedirs('models', exist_ok=True)
    
    df = load_data(input_path)
    df = validate_data(df)
    df = preprocess_text(df)
    X, vectorizer = feature_engineering(df, fit=fit_vectorizer)
    
    print("\n✓ ETL pipeline complete\n")
    return df, X, vectorizer

if __name__ == "__main__":
    df, X, vectorizer = run_etl()
    print(f"Final dataset: {len(df)} tickets, {X.shape[1]} features")
