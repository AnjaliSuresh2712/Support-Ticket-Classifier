import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from etl_pipeline import run_etl
import joblib
import os

def train_and_evaluate_models():
    """Train multiple models and track with MLflow"""
    print("\n" + "="*50)
    print("TRAINING ML MODELS")
    print("="*50 + "\n")
    
    df, X, vectorizer = run_etl()
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    # 2 models for now to compare
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    
    }
    
    best_accuracy = 0
    best_model_name = None
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print('='*50)
        
        with mlflow.start_run(run_name=name):
            #train
            model.fit(X_train, y_train)
            #predict
            y_pred = model.predict(X_test)
            #evaluate
            accuracy = accuracy_score(y_test, y_pred)
            # Log in to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_train", X_train.shape[0])
            mlflow.log_param("n_test", X_test.shape[0])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.1%}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                # Save best model
                joblib.dump(model, 'models/best_model.pkl')
                print(f"\n✓ Saved as best model")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.1%}")
    print("Models saved to models/ directory")
    print("Experiments logged to MLflow")

if __name__ == "__main__":
    train_and_evaluate_models()
