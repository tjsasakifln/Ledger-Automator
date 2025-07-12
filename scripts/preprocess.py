"""
Módulo de pré-processamento para dados de transações financeiras.
Contém funções para limpeza de texto e vetorização.
"""

import re
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Limpa e normaliza texto de descrições de transações.
    
    Args:
        text (str): Texto a ser limpo
        
    Returns:
        str: Texto limpo e normalizado
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text

def preprocess_descriptions(descriptions: List[str]) -> List[str]:
    """
    Aplica limpeza a uma lista de descrições.
    
    Args:
        descriptions (List[str]): Lista de descrições
        
    Returns:
        List[str]: Lista de descrições limpas
    """
    logger.info(f"Pré-processando {len(descriptions)} descrições...")
    
    cleaned_descriptions = [clean_text(desc) for desc in descriptions]
    
    logger.info("Pré-processamento concluído.")
    return cleaned_descriptions

def create_vectorizer(max_features: int = 1000, min_df: int = 1, max_df: float = 0.95) -> TfidfVectorizer:
    """
    Cria um vetorizador TF-IDF configurado.
    
    Args:
        max_features (int): Número máximo de features
        min_df (int): Frequência mínima do documento
        max_df (float): Frequência máxima do documento
        
    Returns:
        TfidfVectorizer: Vetorizador configurado
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=None,  # Não usar stop words em português por simplicidade
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Palavras com pelo menos 2 caracteres
    )
    
    logger.info(f"Vetorizador TF-IDF criado com max_features={max_features}")
    return vectorizer

def vectorize_text(descriptions: List[str], vectorizer: TfidfVectorizer = None, fit: bool = True):
    """
    Vetoriza descrições usando TF-IDF.
    
    Args:
        descriptions (List[str]): Lista de descrições
        vectorizer (TfidfVectorizer, optional): Vetorizador já treinado
        fit (bool): Se deve treinar o vetorizador
        
    Returns:
        Tuple: (matriz_vetorizada, vetorizador)
    """
    try:
        if vectorizer is None:
            vectorizer = create_vectorizer()
        
        cleaned_descriptions = preprocess_descriptions(descriptions)
        
        if fit:
            logger.info("Treinando vetorizador e transformando dados...")
            X_vectorized = vectorizer.fit_transform(cleaned_descriptions)
        else:
            logger.info("Aplicando vetorizador já treinado...")
            X_vectorized = vectorizer.transform(cleaned_descriptions)
        
        logger.info(f"Vetorização concluída. Shape: {X_vectorized.shape}")
        return X_vectorized, vectorizer
        
    except Exception as e:
        logger.error(f"Erro na vetorização: {str(e)}")
        raise

def prepare_training_data(df: pd.DataFrame) -> Tuple:
    """
    Prepara dados de treinamento para o modelo.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'Description' e 'Category'
        
    Returns:
        Tuple: (X_vectorized, y, vectorizer)
    """
    try:
        logger.info("Preparando dados de treinamento...")
        
        if 'Description' not in df.columns or 'Category' not in df.columns:
            raise ValueError("DataFrame deve conter colunas 'Description' e 'Category'")
        
        descriptions = df['Description'].tolist()
        categories = df['Category'].tolist()
        
        X_vectorized, vectorizer = vectorize_text(descriptions, fit=True)
        
        logger.info(f"Dados preparados: {len(descriptions)} amostras, {len(set(categories))} categorias")
        return X_vectorized, categories, vectorizer
        
    except Exception as e:
        logger.error(f"Erro na preparação dos dados: {str(e)}")
        raise

def prepare_prediction_data(df: pd.DataFrame, vectorizer: TfidfVectorizer):
    """
    Prepara dados para predição usando vetorizador já treinado.
    
    Args:
        df (pd.DataFrame): DataFrame com coluna 'Description'
        vectorizer (TfidfVectorizer): Vetorizador já treinado
        
    Returns:
        array: Dados vetorizados para predição
    """
    try:
        logger.info("Preparando dados para predição...")
        
        if 'Description' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'Description'")
        
        descriptions = df['Description'].tolist()
        X_vectorized, _ = vectorize_text(descriptions, vectorizer, fit=False)
        
        logger.info(f"Dados de predição preparados: {X_vectorized.shape}")
        return X_vectorized
        
    except Exception as e:
        logger.error(f"Erro na preparação dos dados de predição: {str(e)}")
        raise

if __name__ == "__main__":
    # Teste das funções
    test_descriptions = [
        "Supermercado Extra Compras",
        "Posto Shell - Combustível",
        "Salário Empresa XYZ",
        "Netflix Streaming 123"
    ]
    
    logger.info("Testando funções de pré-processamento...")
    cleaned = preprocess_descriptions(test_descriptions)
    print("Descrições limpas:", cleaned)
    
    X, vectorizer = vectorize_text(test_descriptions)
    print(f"Shape da matriz vetorizada: {X.shape}")
    print(f"Número de features: {len(vectorizer.get_feature_names_out())}")