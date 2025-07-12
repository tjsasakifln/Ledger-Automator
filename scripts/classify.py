"""
Script para classificação de transações financeiras usando modelo treinado.
Carrega mock_transactions.csv, aplica o modelo e salva resultado em outputs/.
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
    Carrega o modelo treinado e o vetorizador do disco.
    
    Args:
        model_dir (str): Diretório onde estão salvos os arquivos
        
    Returns:
        tuple: (modelo, vetorizador)
    """
    try:
        # Caminhos dos arquivos
        model_path = os.path.join(os.path.dirname(__file__), '..', model_dir, 'model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', model_dir, 'vectorizer.pkl')
        
        # Verificar se arquivos existem
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
        
        logger.info("✅ Modelo e vetorizador carregados com sucesso!")
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

def load_transactions_data():
    """
    Carrega dados de transações do arquivo mock_transactions.csv.
    
    Returns:
        pd.DataFrame: DataFrame com transações
    """
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_transactions.csv')
        logger.info(f"Carregando transações de: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Dataset carregado com {len(df)} transações")
        
        # Verificar colunas obrigatórias
        required_columns = ['Date', 'Description', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar transações: {str(e)}")
        raise

def classify_transactions(df, model, vectorizer):
    """
    Classifica transações usando o modelo treinado.
    
    Args:
        df (pd.DataFrame): DataFrame com transações
        model: Modelo treinado
        vectorizer: Vetorizador TF-IDF treinado
        
    Returns:
        pd.DataFrame: DataFrame com categorias adicionadas
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
    Gera um resumo da classificação realizada.
    
    Args:
        df_classified (pd.DataFrame): DataFrame com transações classificadas
    """
    try:
        print("\n📊 RESUMO DA CLASSIFICAÇÃO:")
        print("=" * 40)
        
        # Contagem por categoria
        category_counts = df_classified['Category'].value_counts()
        print("\n🏷️  TRANSAÇÕES POR CATEGORIA:")
        for category, count in category_counts.items():
            percentage = (count / len(df_classified)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Estatísticas de confiança
        avg_confidence = df_classified['Confidence'].mean()
        min_confidence = df_classified['Confidence'].min()
        print(f"\n🎯 CONFIANÇA MÉDIA: {avg_confidence:.3f}")
        print(f"🎯 CONFIANÇA MÍNIMA: {min_confidence:.3f}")
        
        # Transações com baixa confiança
        low_confidence = df_classified[df_classified['Confidence'] < 0.5]
        if len(low_confidence) > 0:
            print(f"\n⚠️  TRANSAÇÕES COM BAIXA CONFIANÇA ({len(low_confidence)}):")
            for _, row in low_confidence.iterrows():
                print(f"   {row['Description']} -> {row['Category']} ({row['Confidence']:.3f})")
        
    except Exception as e:
        logger.error(f"Erro ao gerar resumo: {str(e)}")

def save_results(df_classified, output_dir="outputs"):
    """
    Salva os resultados da classificação em arquivo CSV.
    
    Args:
        df_classified (pd.DataFrame): DataFrame com transações classificadas
        output_dir (str): Diretório de saída
    """
    try:
        # Criar diretório se não existir
        full_output_dir = os.path.join(os.path.dirname(__file__), '..', output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classified_transactions_{timestamp}.csv"
        output_path = os.path.join(full_output_dir, filename)
        
        # Salvar arquivo principal
        df_classified.to_csv(output_path, index=False)
        logger.info(f"Resultados salvos em: {output_path}")
        
        # Salvar também com nome padrão (para facilitar uso)
        standard_path = os.path.join(full_output_dir, "classified_transactions.csv")
        df_classified.to_csv(standard_path, index=False)
        logger.info(f"Arquivo padrão salvo em: {standard_path}")
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   📄 Com timestamp: {output_path}")
        print(f"   📄 Padrão: {standard_path}")
        
        return output_path, standard_path
        
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {str(e)}")
        raise

def main():
    """
    Função principal para classificação de transações.
    """
    try:
        logger.info("🔍 Iniciando classificação de transações...")
        
        # 1. Carregar modelo e vetorizador
        model, vectorizer = load_model_and_vectorizer()
        
        # 2. Carregar dados de transações
        df_transactions = load_transactions_data()
        
        # 3. Classificar transações
        df_classified = classify_transactions(df_transactions, model, vectorizer)
        
        # 4. Gerar resumo
        generate_classification_summary(df_classified)
        
        # 5. Salvar resultados
        output_path, standard_path = save_results(df_classified)
        
        logger.info("✅ Classificação concluída com sucesso!")
        return df_classified
        
    except Exception as e:
        logger.error(f"❌ Erro na classificação: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("🔍 LEDGER AUTOMATOR - CLASSIFICAÇÃO DE TRANSAÇÕES")
        print("=" * 55)
        
        df_result = main()
        
        print("\n" + "=" * 55)
        print("✅ CLASSIFICAÇÃO CONCLUÍDA!")
        print(f"📈 {len(df_result)} transações classificadas")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("   1. Verifique: outputs/classified_transactions.csv")
        print("   2. Execute: streamlit run scripts/app.py (para visualizar)")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        print("\n💡 DICA: Certifique-se de que o modelo foi treinado primeiro:")
        print("   python scripts/train_model.py")
        exit(1)