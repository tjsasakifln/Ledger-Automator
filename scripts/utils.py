"""
Utility functions for the Ledger Automator.
Contains functions for data loading, file operations, and auxiliary operations.
"""

import pandas as pd
import os
import logging
import joblib
from typing import Union, Tuple, Dict, Any
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str, required_columns: list = None) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV com validação de colunas.
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        required_columns (list, optional): Lista de colunas obrigatórias
        
    Returns:
        pd.DataFrame: DataFrame carregado e validado
    """
    try:
        logger.info(f"Carregando dados de: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Arquivo carregado com {len(df)} linhas e {len(df.columns)} colunas")
        
        # Validar colunas obrigatórias
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
            logger.info("✅ Todas as colunas obrigatórias estão presentes")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

def validate_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e limpa dados de transações financeiras.
    
    Args:
        df (pd.DataFrame): DataFrame com dados de transações
        
    Returns:
        pd.DataFrame: DataFrame validado e limpo
    """
    try:
        logger.info("Validando dados de transações...")
        
        df_clean = df.copy()
        
        # Converter coluna Date
        if 'Date' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
        # Converter coluna Amount
        if 'Amount' in df_clean.columns:
            df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        
        # Limpar Description (remover valores nulos)
        if 'Description' in df_clean.columns:
            df_clean['Description'] = df_clean['Description'].fillna('').astype(str)
        
        # Remover linhas com dados críticos faltando
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Date', 'Amount'])
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            logger.warning(f"Removidas {initial_rows - final_rows} linhas com dados faltando")
        
        logger.info(f"✅ Dados validados: {len(df_clean)} transações válidas")
        return df_clean
        
    except Exception as e:
        logger.error(f"Erro na validação dos dados: {str(e)}")
        raise

def save_classified_data(df: pd.DataFrame, output_path: str, include_timestamp: bool = True) -> str:
    """
    Salva dados classificados em arquivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame com dados classificados
        output_path (str): Caminho de saída
        include_timestamp (bool): Se deve incluir timestamp no nome do arquivo
        
    Returns:
        str: Caminho do arquivo salvo
    """
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Adicionar timestamp se solicitado
        if include_timestamp:
            base_path, ext = os.path.splitext(output_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = f"{base_path}_{timestamp}{ext}"
        else:
            final_path = output_path
        
        # Salvar arquivo
        df.to_csv(final_path, index=False)
        logger.info(f"Dados salvos em: {final_path}")
        
        return final_path
        
    except Exception as e:
        logger.error(f"Erro ao salvar dados: {str(e)}")
        raise

def create_summary_stats(df: pd.DataFrame) -> dict:
    """
    Cria estatísticas resumidas dos dados de transações.
    
    Args:
        df (pd.DataFrame): DataFrame com transações
        
    Returns:
        dict: Dicionário com estatísticas
    """
    try:
        stats = {}
        
        if 'Amount' in df.columns:
            # Estatísticas financeiras
            receitas = df[df['Amount'] > 0]['Amount'].sum()
            despesas = abs(df[df['Amount'] < 0]['Amount'].sum())
            saldo = receitas - despesas
            
            stats['financeiro'] = {
                'total_receitas': receitas,
                'total_despesas': despesas,
                'saldo_liquido': saldo,
                'numero_transacoes': len(df)
            }
        
        if 'Category' in df.columns:
            # Estatísticas por categoria
            category_counts = df['Category'].value_counts().to_dict()
            stats['categorias'] = category_counts
        
        if 'Date' in df.columns:
            # Período dos dados
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            stats['periodo'] = {
                'data_inicio': min_date,
                'data_fim': max_date,
                'dias_cobertura': (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else 0
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Erro ao calcular estatísticas: {str(e)}")
        return {}

def format_currency(value: float, currency: str = "R$") -> str:
    """
    Formata valor como moeda.
    
    Args:
        value (float): Valor a ser formatado
        currency (str): Símbolo da moeda
        
    Returns:
        str: Valor formatado
    """
    try:
        return f"{currency} {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return f"{currency} 0,00"

def ensure_directory_exists(path: str) -> None:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        path (str): Caminho do diretório
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Diretório garantido: {path}")
    except Exception as e:
        logger.error(f"Erro ao criar diretório {path}: {str(e)}")
        raise

def get_project_root() -> str:
    """
    Retorna o diretório raiz do projeto.
    
    Returns:
        str: Caminho do diretório raiz
    """
    # Assume que este arquivo está em scripts/utils.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_path(*args) -> str:
    """
    Constrói um caminho relativo ao projeto.
    
    Args:
        *args: Componentes do caminho
        
    Returns:
        str: Caminho completo
    """
    return os.path.join(get_project_root(), *args)

# Legacy functions for compatibility with existing Streamlit app

def load_csv(file_path):
    """Legacy wrapper for load_data function to maintain app.py compatibility."""
    try:
        if hasattr(file_path, 'read'):  # File upload object
            return pd.read_csv(file_path)
        
        required_columns = ['Date', 'Description', 'Amount']
        df = load_data(file_path, required_columns)
        
        # Convert data types
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'])
        
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def load_model():
    """Load the trained classification model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run the training script first: python scripts/train_model.py"
        )
    
    return joblib.load(model_path)

def load_vectorizer():
    """Load the trained TF-IDF vectorizer."""
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'vectorizer.pkl')
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer not found at {vectorizer_path}. "
            "Please run the training script first: python scripts/train_model.py"
        )
    
    return joblib.load(vectorizer_path)

def classify_transactions(df):
    """Apply automatic classification to transactions."""
    try:
        model = load_model()
        vectorizer = load_vectorizer()
        
        # Import preprocess functions
        import sys
        sys.path.append(os.path.dirname(__file__))
        from preprocess import prepare_prediction_data
        
        # Prepare data and predict
        X_vectorized = prepare_prediction_data(df, vectorizer)
        predicted_categories = model.predict(X_vectorized)
        
        df_classified = df.copy()
        df_classified['Category'] = predicted_categories
        
        return df_classified
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        df_fallback = df.copy()
        df_fallback['Category'] = 'Unclassified'
        return df_fallback

def calculate_monthly_summary(df):
    """Calculate monthly summary of income and expenses."""
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly['Date'].dt.to_period('M')
    
    monthly_summary = df_monthly.groupby('Month').agg({
        'Amount': lambda x: (x[x > 0].sum(), abs(x[x < 0].sum()))
    }).reset_index()
    
    monthly_summary[['Income', 'Expenses']] = pd.DataFrame(
        monthly_summary['Amount'].tolist(), index=monthly_summary.index
    )
    monthly_summary = monthly_summary.drop('Amount', axis=1)
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)
    
    return monthly_summary

def calculate_category_summary(df):
    """Calculate summary by category (expenses only)."""
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = abs(expenses['Amount'])
    
    category_summary = expenses.groupby('Category')['Amount'].sum().reset_index()
    category_summary = category_summary.sort_values('Amount', ascending=False)
    
    return category_summary

def calculate_cumulative_balance(df):
    """Calculate cumulative balance over time."""
    df_sorted = df.sort_values('Date').copy()
    df_sorted['Cumulative_Balance'] = df_sorted['Amount'].cumsum()
    
    return df_sorted[['Date', 'Cumulative_Balance']]

def generate_pl_statement(df, output_path):
    """Generate P&L statement PDF."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        
        title = Paragraph("PROFIT & LOSS STATEMENT", title_style)
        story.append(title)
        
        period_text = f"Period: {df['Date'].min().strftime('%m/%d/%Y')} to {df['Date'].max().strftime('%m/%d/%Y')}"
        period = Paragraph(period_text, styles['Normal'])
        story.append(period)
        story.append(Spacer(1, 20))
        
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        net_profit = total_income - total_expenses
        
        summary_data = [
            ['FINANCIAL SUMMARY', ''],
            ['Total Income', f'$ {total_income:,.2f}'],
            ['Total Expenses', f'$ {total_expenses:,.2f}'],
            ['Net Profit/Loss', f'$ {net_profit:,.2f}']
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        category_title = Paragraph("EXPENSES BY CATEGORY", styles['Heading2'])
        story.append(category_title)
        story.append(Spacer(1, 10))
        
        category_summary = calculate_category_summary(df)
        
        category_data = [['Category', 'Amount ($)']]
        for _, row in category_summary.iterrows():
            category_data.append([row['Category'], f'$ {row["Amount"]:,.2f}'])
        
        category_table = Table(category_data, colWidths=[3*inch, 2*inch])
        category_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(category_table)
        
        footer_text = f"Report generated on {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}"
        footer = Paragraph(footer_text, styles['Normal'])
        story.append(Spacer(1, 30))
        story.append(footer)
        
        doc.build(story)
        return True
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return False

if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing utility functions...")
    
    # Test currency formatting
    print(f"Formatting: {format_currency(1234.56)}")
    
    # Test project paths
    print(f"Project root: {get_project_root()}")
    print(f"Data path: {build_path('data', 'mock_transactions.csv')}")
    
    print("✅ Tests completed!")