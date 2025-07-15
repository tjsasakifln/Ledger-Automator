"""
Controller module for Ledger Automator business logic.
Implements the Controller layer in MVC pattern to separate business logic from UI.
"""

import pandas as pd
import os
import logging
from typing import Optional, Dict, Any, Tuple
from utils import (
    load_csv, 
    classify_transactions, 
    calculate_monthly_summary,
    calculate_category_summary,
    calculate_cumulative_balance,
    generate_pl_statement,
    validate_transaction_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LedgerController:
    """
    Controller class that handles all business logic for the Ledger Automator.
    Separates data processing, model operations, and calculations from the UI layer.
    """
    
    def __init__(self):
        """Initialize the controller."""
        self.current_transactions = None
        self.classified_transactions = None
        
    def load_transaction_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Load and validate transaction data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dict containing success status, data, and message
        """
        try:
            logger.info("Loading transaction file...")
            
            # Load the CSV data
            df = load_csv(uploaded_file)
            
            # Validate the data
            df_validated = validate_transaction_data(df)
            
            # Store in controller state
            self.current_transactions = df_validated
            
            logger.info(f"Successfully loaded {len(df_validated)} transactions")
            
            return {
                'success': True,
                'data': df_validated,
                'message': f"File loaded successfully! {len(df_validated)} transactions found.",
                'row_count': len(df_validated)
            }
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'data': None,
                'message': error_msg,
                'row_count': 0
            }
    
    def load_example_data(self) -> Dict[str, Any]:
        """
        Load example transaction data from mock file.
        
        Returns:
            Dict containing success status, data, and message
        """
        try:
            logger.info("Loading example data...")
            
            example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_transactions.csv')
            df = load_csv(example_path)
            df_validated = validate_transaction_data(df)
            
            # Automatically classify example data
            df_classified = classify_transactions(df_validated)
            
            # Store in controller state
            self.current_transactions = df_validated
            self.classified_transactions = df_classified
            
            logger.info(f"Successfully loaded {len(df_classified)} example transactions")
            
            return {
                'success': True,
                'data': df_classified,
                'message': "Example data loaded and classified successfully!",
                'row_count': len(df_classified)
            }
            
        except Exception as e:
            error_msg = f"Error loading example data: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'data': None,
                'message': error_msg,
                'row_count': 0
            }
    
    def classify_current_transactions(self) -> Dict[str, Any]:
        """
        Classify the currently loaded transactions.
        
        Returns:
            Dict containing success status, data, and message
        """
        try:
            if self.current_transactions is None:
                return {
                    'success': False,
                    'data': None,
                    'message': "No transactions loaded. Please upload a file first."
                }
            
            logger.info("Classifying transactions...")
            
            df_classified = classify_transactions(self.current_transactions)
            self.classified_transactions = df_classified
            
            logger.info(f"Successfully classified {len(df_classified)} transactions")
            
            return {
                'success': True,
                'data': df_classified,
                'message': "Transactions classified successfully!",
                'categories_found': list(df_classified['Category'].unique())
            }
            
        except Exception as e:
            error_msg = f"Error classifying transactions: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'data': None,
                'message': error_msg
            }
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """
        Calculate financial summary metrics.
        
        Returns:
            Dict containing financial metrics
        """
        try:
            if self.classified_transactions is None:
                return {
                    'success': False,
                    'message': "No classified transactions available."
                }
            
            df = self.classified_transactions
            
            total_income = df[df['Amount'] > 0]['Amount'].sum()
            total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
            net_balance = total_income - total_expenses
            transaction_count = len(df)
            
            return {
                'success': True,
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_balance': net_balance,
                'transaction_count': transaction_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating financial summary: {str(e)}")
            return {
                'success': False,
                'message': f"Error calculating summary: {str(e)}"
            }
    
    def get_chart_data(self) -> Dict[str, Any]:
        """
        Get data for dashboard charts.
        
        Returns:
            Dict containing chart data
        """
        try:
            if self.classified_transactions is None:
                return {
                    'success': False,
                    'message': "No classified transactions available."
                }
            
            df = self.classified_transactions
            
            # Category summary for pie chart
            category_summary = calculate_category_summary(df)
            
            # Monthly summary for bar chart
            monthly_summary = calculate_monthly_summary(df)
            
            # Cumulative balance for line chart
            cumulative_data = calculate_cumulative_balance(df)
            
            return {
                'success': True,
                'category_summary': category_summary,
                'monthly_summary': monthly_summary,
                'cumulative_data': cumulative_data
            }
            
        except Exception as e:
            logger.error(f"Error getting chart data: {str(e)}")
            return {
                'success': False,
                'message': f"Error generating chart data: {str(e)}"
            }
    
    def filter_transactions_by_category(self, category: str) -> pd.DataFrame:
        """
        Filter transactions by category.
        
        Args:
            category: Category to filter by ('All' for no filter)
            
        Returns:
            Filtered DataFrame
        """
        if self.classified_transactions is None:
            return pd.DataFrame()
        
        if category == 'All':
            return self.classified_transactions
        else:
            return self.classified_transactions[self.classified_transactions['Category'] == category]
    
    def generate_pdf_statement(self, output_filename: str = "p&l_statement.pdf") -> Dict[str, Any]:
        """
        Generate PDF P&L statement.
        
        Args:
            output_filename: Name of the output PDF file
            
        Returns:
            Dict containing success status and file path
        """
        try:
            if self.classified_transactions is None:
                return {
                    'success': False,
                    'message': "No classified transactions available."
                }
            
            output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', output_filename)
            
            logger.info(f"Generating PDF statement: {output_path}")
            
            success = generate_pl_statement(self.classified_transactions, output_path)
            
            if success:
                return {
                    'success': True,
                    'file_path': output_path,
                    'message': "PDF statement generated successfully!"
                }
            else:
                return {
                    'success': False,
                    'message': "Failed to generate PDF statement."
                }
                
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return {
                'success': False,
                'message': f"Error generating PDF: {str(e)}"
            }
    
    def get_available_categories(self) -> list:
        """
        Get list of available categories for filtering.
        
        Returns:
            List of category names
        """
        if self.classified_transactions is None:
            return []
        
        return ['All'] + list(self.classified_transactions['Category'].unique())
    
    def has_transactions(self) -> bool:
        """Check if transactions are loaded."""
        return self.current_transactions is not None
    
    def has_classified_transactions(self) -> bool:
        """Check if classified transactions are available."""
        return self.classified_transactions is not None
    
    def get_current_transactions(self) -> Optional[pd.DataFrame]:
        """Get current transactions DataFrame."""
        return self.current_transactions
    
    def get_classified_transactions(self) -> Optional[pd.DataFrame]:
        """Get classified transactions DataFrame."""
        return self.classified_transactions