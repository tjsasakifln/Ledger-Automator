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

def clean_text(text: str, preserve_numbers: bool = False) -> str:
    """
    Clean and normalize transaction description text with intelligent preprocessing.
    
    Args:
        text (str): Text to be cleaned
        preserve_numbers (bool): Whether to preserve numerical information
        
    Returns:
        str: Cleaned and normalized text
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Preserve important financial keywords before cleaning
    financial_patterns = {
        'credit': ['credit', 'credito', 'cr'],
        'debit': ['debit', 'debito', 'db'],
        'payment': ['payment', 'pagamento', 'pag'],
        'transfer': ['transfer', 'transferencia', 'transf'],
        'withdrawal': ['withdrawal', 'saque'],
        'deposit': ['deposit', 'deposito']
    }
    
    # Normalize financial terms
    for standard_term, variations in financial_patterns.items():
        for variation in variations:
            text = re.sub(rf'\b{variation}\b', standard_term, text)
    
    # Remove special characters but preserve spaces and hyphens in names
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Conditionally remove or preserve numbers
    if not preserve_numbers:
        # Remove standalone numbers but preserve numbers that are part of names
        text = re.sub(r'\b\d+\b', '', text)
    
    # Clean up multiple spaces
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

def get_financial_stop_words() -> list:
    """
    Get comprehensive stop words for financial text processing.
    
    Returns:
        list: Combined English and Portuguese stop words for financial context
    """
    english_stop_words = [
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with',
        'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during',
        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
        'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves'
    ]
    
    portuguese_stop_words = [
        'de', 'da', 'do', 'das', 'dos', 'a', 'o', 'as', 'os', 'e', 'ou', 'mas', 'em',
        'com', 'para', 'por', 'sobre', 'ate', 'desde', 'durante', 'antes', 'depois',
        'acima', 'abaixo', 'entre', 'atraves', 'este', 'esta', 'estes', 'estas',
        'esse', 'essa', 'esses', 'essas', 'aquele', 'aquela', 'aqueles', 'aquelas',
        'eu', 'me', 'meu', 'minha', 'meus', 'minhas', 'nos', 'nosso', 'nossa',
        'nossos', 'nossas', 'voce', 'seu', 'sua', 'seus', 'suas', 'ele', 'ela',
        'eles', 'elas', 'se', 'si', 'mesmo', 'mesma', 'mesmos', 'mesmas'
    ]
    
    # Exclude important financial terms from stop words
    financial_keep_words = [
        'bank', 'banco', 'card', 'cartao', 'credit', 'credito', 'debit', 'debito',
        'payment', 'pagamento', 'transfer', 'transferencia', 'cash', 'dinheiro'
    ]
    
    combined_stop_words = list(set(english_stop_words + portuguese_stop_words))
    # Remove financial terms from stop words
    combined_stop_words = [word for word in combined_stop_words if word not in financial_keep_words]
    
    return combined_stop_words

def create_vectorizer(max_features: int = 1000, min_df: int = 1, max_df: float = 0.95) -> TfidfVectorizer:
    """
    Create configured TF-IDF vectorizer with intelligent stop words.
    
    Args:
        max_features (int): Maximum number of features
        min_df (int): Minimum document frequency
        max_df (float): Maximum document frequency
        
    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    stop_words = get_financial_stop_words()
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Words with at least 2 characters
        ngram_range=(1, 2)  # Include bigrams for better context
    )
    
    logger.info(f"TF-IDF vectorizer created with max_features={max_features}, stop_words={len(stop_words)}")
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