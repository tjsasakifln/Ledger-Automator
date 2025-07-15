# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class AppConfig:
    """Centralized configuration"""
    APP_TITLE: str = "Ledger Automator"
    APP_ICON: str = "📊"
    SUPPORTED_CATEGORIES: List[str] = None
    MODEL_PATH: str = "outputs/model.pkl"
    VECTORIZER_PATH: str = "outputs/vectorizer.pkl"
    MIN_CONFIDENCE_THRESHOLD: float = 0.5
    
    def __post_init__(self):
        if self.SUPPORTED_CATEGORIES is None:
            self.SUPPORTED_CATEGORIES = [
                "Food", "Transportation", "Income", "Healthcare",
                "Utilities", "Entertainment", "Housing", "Shopping"
            ]

# services.py
from typing import Optional
import pandas as pd
import joblib
from config import AppConfig

class ModelService:
    """Handles ML model operations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._vectorizer = None
    
    def load_model(self) -> bool:
        """Load model and vectorizer"""
        try:
            self._model = joblib.load(self.config.MODEL_PATH)
            self._vectorizer = joblib.load(self.config.VECTORIZER_PATH)
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    
    def classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify transactions with error handling"""
        if not self._model or not self._vectorizer:
            if not self.load_model():
                # Fallback classification
                df_fallback = df.copy()
                df_fallback['Category'] = 'Unclassified'
                df_fallback['Confidence'] = 0.0
                return df_fallback
        
        try:
            from preprocess import prepare_prediction_data
            X_vectorized = prepare_prediction_data(df, self._vectorizer)
            predictions = self._model.predict(X_vectorized)
            probabilities = self._model.predict_proba(X_vectorized)
            confidence_scores = probabilities.max(axis=1)
            
            df_classified = df.copy()
            df_classified['Category'] = predictions
            df_classified['Confidence'] = confidence_scores
            
            return df_classified
            
        except Exception as e:
            st.error(f"Classification failed: {e}")
            return self._fallback_classification(df)
    
    def _fallback_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple rule-based fallback"""
        df_fallback = df.copy()
        
        # Simple keyword-based classification
        def classify_by_keywords(description: str) -> str:
            desc_lower = str(description).lower()
            
            food_keywords = ['supermarket', 'restaurant', 'food', 'grocery']
            transport_keywords = ['gas', 'fuel', 'uber', 'taxi', 'bus']
            
            if any(kw in desc_lower for kw in food_keywords):
                return 'Food'
            elif any(kw in desc_lower for kw in transport_keywords):
                return 'Transportation'
            else:
                return 'Other'
        
        df_fallback['Category'] = df_fallback['Description'].apply(classify_by_keywords)
        df_fallback['Confidence'] = 0.3  # Low confidence for rule-based
        
        return df_fallback

class DataService:
    """Handles data operations"""
    
    @staticmethod
    def validate_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean transaction data"""
        required_columns = ['Date', 'Description', 'Amount']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_clean = df.copy()
        
        # Convert and validate data types
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        df_clean['Description'] = df_clean['Description'].fillna('').astype(str)
        
        # Remove rows with invalid data
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['Date', 'Amount'])
        after_count = len(df_clean)
        
        if before_count != after_count:
            st.warning(f"Removed {before_count - after_count} rows with invalid data")
        
        return df_clean
    
    @staticmethod
    def calculate_financial_summary(df: pd.DataFrame) -> dict:
        """Calculate financial summary statistics"""
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        net_balance = total_income - total_expenses
        transaction_count = len(df)
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_balance': net_balance,
            'transaction_count': transaction_count
        }

# components.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_financial_metrics(summary: dict):
        """Render financial metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Income", f"${summary['total_income']:,.2f}")
        with col2:
            st.metric("💸 Total Expenses", f"${summary['total_expenses']:,.2f}")
        with col3:
            st.metric("📊 Net Balance", f"${summary['net_balance']:,.2f}")
        with col4:
            st.metric("🔢 Transactions", summary['transaction_count'])
    
    @staticmethod
    def render_category_pie_chart(df: pd.DataFrame):
        """Render expenses by category pie chart"""
        expenses = df[df['Amount'] < 0].copy()
        if expenses.empty:
            st.info("No expenses found to categorize")
            return
        
        expenses['Amount'] = abs(expenses['Amount'])
        category_summary = expenses.groupby('Category')['Amount'].sum().reset_index()
        
        fig = px.pie(
            category_summary,
            values='Amount',
            names='Category',
            title="Expenses by Category"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_file_uploader():
        """Render file upload component with validation"""
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File must contain columns: Date, Description, Amount"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded successfully! {len(df)} transactions found.")
                
                # Show preview
                st.subheader("Data Preview:")
                st.dataframe(df.head(10))
                
                return df
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                return None
        
        return None

# app_refactored.py
import streamlit as st
from config import AppConfig
from services import ModelService, DataService
from components import UIComponents

class LedgerAutomatorApp:
    """Main application class"""
    
    def __init__(self):
        self.config = AppConfig()
        self.model_service = ModelService(self.config)
        self.data_service = DataService()
        self.ui = UIComponents()
        
        # Configure Streamlit
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            page_icon=self.config.APP_ICON,
            layout="wide"
        )
    
    def run(self):
        """Main application entry point"""
        st.title(f"{self.config.APP_ICON} {self.config.APP_TITLE}")
        st.markdown("Automatic Financial Transaction Classification System")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose an option:",
            ["Data Upload", "Dashboard", "P&L Statement"]
        )
        
        # Route to appropriate page
        if page == "Data Upload":
            self._render_upload_page()
        elif page == "Dashboard":
            self._render_dashboard_page()
        else:
            self._render_pl_page()
    
    def _render_upload_page(self):
        """Render the data upload page"""
        st.header("📂 CSV File Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Expected CSV format:**
            - Required columns: `Date`, `Description`, `Amount`
            - Date: YYYY-MM-DD format
            - Description: transaction description
            - Amount: value (positive for income, negative for expenses)
            """)
            
            df = self.ui.render_file_uploader()
            
            if df is not None:
                if st.button("🤖 Classify Transactions", type="primary"):
                    with st.spinner("Classifying transactions..."):
                        try:
                            df_validated = self.data_service.validate_transaction_data(df)
                            df_classified = self.model_service.classify_transactions(df_validated)
                            st.session_state['transactions'] = df_classified
                            st.success("✅ Transactions classified successfully!")
                            st.dataframe(df_classified.head(10))
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.markdown("### 📄 Sample Data")
            if st.button("Load sample data"):
                # Load sample data logic here
                pass
    
    def _render_dashboard_page(self):
        """Render the dashboard page"""
        st.header("📈 Financial Dashboard")
        
        if 'transactions' not in st.session_state:
            st.warning("⚠️ Please upload a file first in the 'Data Upload' page")
            return
        
        df = st.session_state['transactions']
        
        # Financial summary
        summary = self.data_service.calculate_financial_summary(df)
        self.ui.render_financial_metrics(summary)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍰 Expenses by Category")
            self.ui.render_category_pie_chart(df)
        
        with col2:
            st.subheader("📊 Monthly Trends")
            # Monthly trends chart implementation
            pass
        
        # Transaction table
        st.subheader("📋 Transaction Table")
        
        # Filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            categories = ['All'] + list(df['Category'].unique())
            selected_category = st.selectbox("Filter by category:", categories)
        
        # Apply filter
        filtered_df = df if selected_category == 'All' else df[df['Category'] == selected_category]
        
        st.dataframe(
            filtered_df.sort_values('Date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    def _render_pl_page(self):
        """Render the P&L statement page"""
        st.header("📄 Profit & Loss Statement")
        
        if 'transactions' not in st.session_state:
            st.warning("⚠️ Please upload a file first in the 'Data Upload' page")
            return
        
        # P&L implementation here
        pass

if __name__ == "__main__":
    app = LedgerAutomatorApp()
    app.run()