# ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, accuracy_score, 
    confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Enhanced ML training pipeline with proper validation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = 0
        self.models_to_try = {
            'logistic_regression': LogisticRegression(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'naive_bayes': MultinomialNB()
        }
    
    def validate_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate training data quality"""
        logger.info("Validating training data...")
        
        # Check required columns
        required_cols = ['Description', 'Category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove empty descriptions
        initial_count = len(df)
        df = df.dropna(subset=['Description'])
        df = df[df['Description'].str.strip() != '']
        final_count = len(df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with empty descriptions")
        
        # Check category distribution
        category_counts = df['Category'].value_counts()
        logger.info(f"Category distribution:\n{category_counts}")
        
        # Warn about imbalanced classes
        min_samples = category_counts.min()
        if min_samples < 3:
            logger.warning(f"Some categories have very few samples (min: {min_samples})")
            logger.warning("Consider collecting more data for better performance")
        
        # Check for minimum data requirements
        if len(df) < 50:
            logger.warning("Dataset is very small (<50 samples). Results may be unreliable.")
        
        return df
    
    def prepare_data_with_validation(self, df: pd.DataFrame) -> Tuple:
        """Prepare data with enhanced preprocessing"""
        from preprocess import prepare_training_data
        
        # Validate data first
        df_validated = self.validate_training_data(df)
        
        # Prepare features and labels
        X_vectorized, y, vectorizer = prepare_training_data(df_validated)
        
        # Check for class imbalance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        
        logger.info(f"Class distribution: {class_distribution}")
        
        # Calculate class weights for imbalanced data
        class_weights = {}
        for class_name, count in class_distribution.items():
            class_weights[class_name] = len(y) / (len(unique_classes) * count)
        
        return X_vectorized, y, vectorizer, class_weights
    
    def train_with_cross_validation(self, X, y, vectorizer, class_weights: dict):
        """Train multiple models with cross-validation"""
        logger.info("Training models with cross-validation...")
        
        # Use stratified k-fold to maintain class distribution
        cv_folds = min(5, min([count for count in 
                              pd.Series(y).value_counts().values]))
        
        if cv_folds < 3:
            logger.warning("Not enough data for proper cross-validation")
            cv_folds = 2
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.random_state)
        
        model_scores = {}
        
        for model_name, model in self.models_to_try.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Set class weights if model supports it
                if hasattr(model, 'class_weight'):
                    model.set_params(class_weight=class_weights)
                
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, 
                                       scoring='f1_macro', n_jobs=-1)
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                model_scores[model_name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores
                }
                
                logger.info(f"{model_name} - Mean F1: {mean_score:.3f} (+/- {std_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select best model
        if model_scores:
            best_model_name = max(model_scores.keys(), 
                                key=lambda x: model_scores[x]['mean_score'])
            
            self.best_score = model_scores[best_model_name]['mean_score']
            logger.info(f"Best model: {best_model_name} (F1: {self.best_score:.3f})")
            
            return best_model_name, model_scores
        else:
            raise RuntimeError("No models could be trained successfully")
    
    def hyperparameter_tuning(self, X, y, best_model_name: str, class_weights: dict):
        """Perform hyperparameter tuning on the best model"""
        logger.info(f"Tuning hyperparameters for {best_model_name}...")
        
        # Define parameter grids for each model
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'naive_bayes': {
                'alpha': [0.1, 1.0, 10.0]
            }
        }
        
        if best_model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {best_model_name}")
            return self.models_to_try[best_model_name]
        
        # Set up model with class weights
        model = self.models_to_try[best_model_name]
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight=class_weights)
        
        # Perform grid search
        try:
            cv_folds = min(3, min([count for count in 
                                  pd.Series(y).value_counts().values]))
            
            grid_search = GridSearchCV(
                model, 
                param_grids[best_model_name],
                cv=cv_folds,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            return model
    
    def final_evaluation(self, model, X, y) -> Dict[str, Any]:
        """Perform final evaluation on the trained model"""
        logger.info("Performing final evaluation...")
        
        # If we have enough data, do a proper train/test split
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.random_state
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            evaluation_results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'test_size': len(y_test)
            }
            
            logger.info(f"Final accuracy: {accuracy:.3f}")
            logger.info(f"Final F1 score: {f1:.3f}")
            
        else:
            # For small datasets, just train on all data
            logger.warning("Dataset too small for train/test split. Training on all data.")
            model.fit(X, y)
            
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            evaluation_results = {
                'accuracy': accuracy,
                'f1_score': accuracy,  # Same as accuracy for perfect fit
                'classification_report': classification_report(y, y_pred),
                'note': 'Trained on full dataset due to small size'
            }
        
        return evaluation_results
    
    def train_full_pipeline(self, df: pd.DataFrame) -> Tuple:
        """Complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        try:
            # 1. Prepare and validate data
            X, y, vectorizer, class_weights = self.prepare_data_with_validation(df)
            
            # 2. Train models with cross-validation
            best_model_name, model_scores = self.train_with_cross_validation(
                X, y, vectorizer, class_weights
            )
            
            # 3. Hyperparameter tuning
            tuned_model = self.hyperparameter_tuning(
                X, y, best_model_name, class_weights
            )
            
            # 4. Final evaluation
            evaluation_results = self.final_evaluation(tuned_model, X, y)
            
            # 5. Store best model and vectorizer
            self.best_model = tuned_model
            self.best_vectorizer = vectorizer
            
            logger.info("✅ Training pipeline completed successfully!")
            
            return (self.best_model, self.best_vectorizer, 
                   evaluation_results, model_scores)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def save_model(self, output_dir: str = "outputs"):
        """Save the trained model and vectorizer"""
        if not self.best_model or not self.best_vectorizer:
            raise ValueError("No trained model to save")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'model.pkl')
        vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.best_vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Vectorizer saved to: {vectorizer_path}")
        
        return model_path, vectorizer_path

# enhanced_train_model.py
"""
Enhanced training script with proper ML practices
"""

import os
import pandas as pd
import logging
from ml_pipeline import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_augment_training_data():
    """Load training data and suggest augmentation strategies"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} training samples")
    
    # Analyze data quality
    category_counts = df['Category'].value_counts()
    logger.info(f"Categories: {category_counts.to_dict()}")
    
    # Suggest improvements
    min_samples = category_counts.min()
    if min_samples < 10:
        logger.warning(f"""
        DATA QUALITY WARNING:
        - Minimum samples per category: {min_samples}
        - Recommended minimum: 20-50 samples per category
        - Current total: {len(df)} samples
        - Recommended total: 200+ samples
        
        SUGGESTIONS:
        1. Add more real transaction data
        2. Use data augmentation techniques
        3. Consider combining similar categories
        """)
    
    return df

def generate_training_report(evaluation_results, model_scores, output_dir="outputs"):
    """Generate a comprehensive training report"""
    import json
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_performance': evaluation_results,
        'cross_validation_scores': {
            name: {
                'mean_f1': float(scores['mean_score']),
                'std_f1': float(scores['std_score'])
            } for name, scores in model_scores.items()
        },
        'recommendations': []
    }
    
    # Add recommendations based on performance
    if evaluation_results['accuracy'] < 0.7:
        report['recommendations'].append(
            "Low accuracy detected. Consider collecting more training data."
        )
    
    if evaluation_results.get('test_size', 0) < 10:
        report['recommendations'].append(
            "Test set is very small. Results may not be reliable."
        )
    
    # Save report
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to: {report_path}")
    return report

def main():
    """Enhanced main training function"""
    try:
        print("🚀 ENHANCED LEDGER AUTOMATOR TRAINING")
        print("=" * 50)
        
        # 1. Load and analyze data
        df = load_and_augment_training_data()
        
        # 2. Initialize enhanced trainer
        trainer = ModelTrainer(random_state=42)
        
        # 3. Run complete training pipeline
        model, vectorizer, evaluation_results, model_scores = trainer.train_full_pipeline(df)
        
        # 4. Save model
        model_path, vectorizer_path = trainer.save_model()
        
        # 5. Generate report
        report = generate_training_report(evaluation_results, model_scores)
        
        # 6. Print summary
        print("\n" + "=" * 50)
        print("✅ ENHANCED TRAINING COMPLETED!")
        print(f"📈 Final Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"📈 F1 Score: {evaluation_results['f1_score']:.3f}")
        print(f"💾 Model: {model_path}")
        print(f"💾 Vectorizer: {vectorizer_path}")
        
        print("\n📋 NEXT STEPS:")
        print("1. Review training_report.json for detailed analysis")
        print("2. Run: python scripts/classify.py")
        print("3. Test with: streamlit run scripts/app.py")
        
        if evaluation_results['accuracy'] < 0.8:
            print("\n⚠️  WARNING: Model accuracy is below 80%")
            print("   Consider collecting more training data for better performance")
        
        return model, vectorizer, evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()