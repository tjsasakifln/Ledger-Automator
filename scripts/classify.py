"""
Script for financial transaction classification using trained model.
Loads mock_transactions.csv, applies model and saves results to outputs/.
"""

import pandas as pd
import joblib
import os
import logging
import sys
from datetime import datetime

# Adicionar o diretório scripts ao path para importar módulos locais
sys.path.append(os.path.dirname(__file__))
from preprocess import prepare_prediction_data
from utils import load_data, save_classified_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_vectorizer(model_dir="outputs"):
    """
    Loads the trained model and vectorizer from disk.
    
    Args:
        model_dir (str): Directory where files are saved
        
    Returns:
        tuple: (model, vectorizer)
    """
    try:
        # Caminhos dos arquivos
        model_path = os.path.join(os.path.dirname(__file__), '..', model_dir, 'model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', model_dir, 'vectorizer.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vetorizador não encontrado em: {vectorizer_path}")
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Load vectorizer
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        
        logger.info("✅ Model and vectorizer loaded successfully!")
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_transactions_data():
    """
    Loads transaction data from mock_transactions.csv file.
    
    Returns:
        pd.DataFrame: DataFrame with transactions
    """
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_transactions.csv')
        logger.info(f"Carregando transações de: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Dataset carregado com {len(df)} transações")
        
        # Check required columns
        required_columns = ['Date', 'Description', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading transactions: {str(e)}")
        raise

def classify_transactions(df, model, vectorizer):
    """
    Classifies transactions using the trained model.
    
    Args:
        df (pd.DataFrame): DataFrame with transactions
        model: Trained model
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        pd.DataFrame: DataFrame with added categories
    """
    try:
        logger.info("Iniciando classificação das transações...")
        
        # Preparar dados para predição
        X_vectorized = prepare_prediction_data(df, vectorizer)
        
        # Fazer predições
        logger.info("Aplicando modelo...")
        predictions = model.predict(X_vectorized)
        
        # Obter probabilidades (confiança)
        try:
            probabilities = model.predict_proba(X_vectorized)
            confidence_scores = probabilities.max(axis=1)
        except Exception:
            # Se o modelo não suportar predict_proba
            confidence_scores = [1.0] * len(predictions)
        
        # Adicionar resultados ao DataFrame
        df_classified = df.copy()
        df_classified['Category'] = predictions
        df_classified['Confidence'] = confidence_scores
        
        logger.info(f"✅ Classificação concluída!")
        logger.info(f"Categorias encontradas: {set(predictions)}")
        
        return df_classified
        
    except Exception as e:
        logger.error(f"Erro na classificação: {str(e)}")
        raise

def generate_classification_summary(df_classified):
    """
    Generates a summary of the classification performed.
    
    Args:
        df_classified (pd.DataFrame): DataFrame with classified transactions
    """
    try:
        print("\n📊 CLASSIFICATION SUMMARY:")
        print("=" * 40)
        
        # Count by category
        category_counts = df_classified['Category'].value_counts()
        print("\n🏷️  TRANSACTIONS BY CATEGORY:")
        for category, count in category_counts.items():
            percentage = (count / len(df_classified)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Confidence statistics
        avg_confidence = df_classified['Confidence'].mean()
        min_confidence = df_classified['Confidence'].min()
        print(f"\n🎯 AVERAGE CONFIDENCE: {avg_confidence:.3f}")
        print(f"🎯 MINIMUM CONFIDENCE: {min_confidence:.3f}")
        
        # Low confidence transactions
        low_confidence = df_classified[df_classified['Confidence'] < 0.5]
        if len(low_confidence) > 0:
            print(f"\n⚠️  LOW CONFIDENCE TRANSACTIONS ({len(low_confidence)}):")
            for _, row in low_confidence.iterrows():
                print(f"   {row['Description']} -> {row['Category']} ({row['Confidence']:.3f})")
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")

def save_results(df_classified, output_dir="outputs"):
    """
    Saves classification results to CSV file.
    
    Args:
        df_classified (pd.DataFrame): DataFrame with classified transactions
        output_dir (str): Output directory
    """
    try:
        # Create directory if it does not exist
        full_output_dir = os.path.join(os.path.dirname(__file__), '..', output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classified_transactions_{timestamp}.csv"
        output_path = os.path.join(full_output_dir, filename)
        
        # Save main file
        df_classified.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        # Also save with standard name (for easier use)
        standard_path = os.path.join(full_output_dir, "classified_transactions.csv")
        df_classified.to_csv(standard_path, index=False)
        logger.info(f"Standard file saved to: {standard_path}")
        
        print(f"\n💾 FILES SAVED:")
        print(f"   📄 With timestamp: {output_path}")
        print(f"   📄 Standard: {standard_path}")
        
        return output_path, standard_path
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    """
    Main function for transaction classification.
    """
    try:
        logger.info("🔍 Starting transaction classification...")
        
        # 1. Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer()
        
        # 2. Load transaction data
        df_transactions = load_transactions_data()
        
        # 3. Classify transactions
        df_classified = classify_transactions(df_transactions, model, vectorizer)
        
        # 4. Generate summary
        generate_classification_summary(df_classified)
        
        # 5. Save results
        output_path, standard_path = save_results(df_classified)
        
        logger.info("✅ Classification completed successfully!")
        return df_classified
        
    except Exception as e:
        logger.error(f"❌ Error in classification: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("🔍 LEDGER AUTOMATOR - TRANSACTION CLASSIFICATION")
        print("=" * 55)
        
        df_result = main()
        
        print("\n" + "=" * 55)
        print("✅ CLASSIFICATION COMPLETED!")
        print(f"📈 {len(df_result)} transactions classified")
        print("\n📋 NEXT STEPS:")
        print("   1. Check: outputs/classified_transactions.csv")
        print("   2. Run: streamlit run scripts/app.py (to visualize)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n💡 TIP: Make sure the model was trained first:")
        print("   python scripts/train_model.py")
        exit(1)