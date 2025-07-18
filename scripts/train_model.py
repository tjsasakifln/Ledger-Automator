"""
Training script for financial transaction classification model.
Uses training_data.csv to train a machine learning model.
"""

import pandas as pd
import joblib
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys

# Add scripts directory to path for importing local modules
sys.path.append(os.path.dirname(__file__))
from preprocess import prepare_training_data
from data_augmentation import FinancialDataAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data():
    """
    Load training data from CSV file and generate additional synthetic data.
    
    Returns:
        pd.DataFrame: DataFrame with training data (original + synthetic)
    """
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv')
        logger.info(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        # Load original data
        df_original = pd.read_csv(data_path)
        logger.info(f"Original dataset loaded with {len(df_original)} samples")
        
        # Generate synthetic data for training enhancement
        logger.info("Generating synthetic training data...")
        augmenter = FinancialDataAugmenter()
        df_synthetic = augmenter.generate_training_data()
        logger.info(f"Generated {len(df_synthetic)} synthetic samples")
        
        # Combine original and synthetic data
        df_combined = pd.concat([df_original, df_synthetic], ignore_index=True)
        logger.info(f"Combined dataset: {len(df_combined)} total samples")
        logger.info(f"Unique categories: {df_combined['Category'].unique()}")
        
        # Show distribution by category
        category_counts = df_combined['Category'].value_counts()
        logger.info("Category distribution:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} samples")
        
        return df_combined
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_classifier():
    """
    Creates a LogisticRegression classifier.
    
    Returns:
        LogisticRegression: Configured model
    """
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,
        solver='liblinear'
    )
    
    logger.info("LogisticRegression classifier created")
    return classifier

def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the performance of the trained model.
    
    Args:
        classifier: Trained model
        X_test: Test data (features)
        y_test: Test labels
    """
    try:
        logger.info("Evaluating model...")
        
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        print(f"\n🎯 ACCURACY: {accuracy:.3f}")
        print("\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

def save_model_and_vectorizer(classifier, vectorizer, model_dir="outputs"):
    """
    Saves the trained model and vectorizer to disk.
    
    Args:
        classifier: Trained model
        vectorizer: Trained TF-IDF vectorizer
        model_dir (str): Directory to save files
    """
    try:
        # Create directory if it does not exist
        full_model_dir = os.path.join(os.path.dirname(__file__), '..', model_dir)
        os.makedirs(full_model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(full_model_dir, 'model.pkl')
        joblib.dump(classifier, model_path)
        logger.info(f"Model saved at: {model_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(full_model_dir, 'vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved at: {vectorizer_path}")
        
        print(f"\n💾 FILES SAVED:")
        print(f"   📄 Model: {model_path}")
        print(f"   📄 Vectorizer: {vectorizer_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def train_model():
    """
    Main function for training the classification model.
    """
    try:
        logger.info("🚀 Starting model training...")
        
        # 1. Load data
        df = load_training_data()
        
        # 2. Prepare data (preprocessing + vectorization)
        X_vectorized, y, vectorizer = prepare_training_data(df)
        
        # 3. Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # 4. Create and train model
        classifier = create_classifier()
        logger.info("Training model...")
        classifier.fit(X_train, y_train)
        
        # 5. Evaluate model
        accuracy = evaluate_model(classifier, X_test, y_test)
        
        # 6. Save model and vectorizer
        save_model_and_vectorizer(classifier, vectorizer)
        
        logger.info("✅ Training completed successfully!")
        return classifier, vectorizer, accuracy
        
    except Exception as e:
        logger.error(f"❌ Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("🔬 LEDGER AUTOMATOR - MODEL TRAINING")
        print("=" * 50)
        
        classifier, vectorizer, accuracy = train_model()
        
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETED!")
        print(f"📈 Final accuracy: {accuracy:.3f}")
        print("\n📋 NEXT STEPS:")
        print("   1. Run: python scripts/classify.py")
        print("   2. Check file: outputs/classified_transactions.csv")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        exit(1)