"""
Views module for Ledger Automator UI components.
Implements the View layer in MVC pattern to separate UI rendering from business logic.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Dict, Any


class LedgerViews:
    """
    View class that handles all UI rendering for the Ledger Automator.
    Separates UI components and layout from business logic.
    """
    
    @staticmethod
    def render_page_header():
        """Render the main page header and configuration."""
        st.set_page_config(
            page_title="Ledger Automator",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Ledger Automator")
        st.markdown("Automated financial transaction classification system")
    
    @staticmethod
    def render_sidebar_navigation():
        """
        Render sidebar navigation.
        
        Returns:
            Selected page name
        """
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose an option:",
            ["Data Upload", "Dashboard", "P&L Statement"]
        )
        return page
    
    @staticmethod
    def render_file_upload_section():
        """
        Render file upload section.
        
        Returns:
            Tuple of (uploaded_file, load_example_clicked)
        """
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Expected CSV file format:**
            - Required columns: `Date`, `Description`, `Amount`
            - Date: YYYY-MM-DD format
            - Description: transaction description
            - Amount: value (positive for income, negative for expenses)
            """)
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="File must contain columns: Date, Description, Amount"
            )
        
        with col2:
            st.markdown("### 📄 Example Data")
            load_example = st.button("Load example data")
        
        return uploaded_file, load_example
    
    @staticmethod
    def render_data_preview(df: pd.DataFrame, title: str = "Data preview:"):
        """
        Render data preview table.
        
        Args:
            df: DataFrame to display
            title: Section title
        """
        st.subheader(title)
        st.dataframe(df.head(10))
    
    @staticmethod
    def render_classification_button():
        """
        Render classification button.
        
        Returns:
            True if button was clicked
        """
        return st.button("🤖 Classify Transactions", type="primary")
    
    @staticmethod
    def render_success_message(message: str):
        """Render success message."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def render_error_message(message: str):
        """Render error message."""
        st.error(f"❌ {message}")
    
    @staticmethod
    def render_warning_message(message: str):
        """Render warning message."""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def render_info_message(message: str):
        """Render info message."""
        st.info(message)
    
    @staticmethod
    def render_loading_spinner(message: str):
        """
        Context manager for loading spinner.
        
        Args:
            message: Loading message
        """
        return st.spinner(message)
    
    @staticmethod
    def render_financial_metrics(metrics: Dict[str, Any]):
        """
        Render financial metrics cards.
        
        Args:
            metrics: Dictionary containing financial metrics
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Income", f"$ {metrics['total_income']:,.2f}")
        with col2:
            st.metric("💸 Total Expenses", f"$ {metrics['total_expenses']:,.2f}")
        with col3:
            st.metric("📊 Net Balance", f"$ {metrics['net_balance']:,.2f}")
        with col4:
            st.metric("🔢 Transactions", metrics['transaction_count'])
    
    @staticmethod
    def render_category_pie_chart(category_data: pd.DataFrame):
        """
        Render category expenses pie chart.
        
        Args:
            category_data: DataFrame with category summary
        """
        st.subheader("🍰 Expenses by Category")
        
        if not category_data.empty:
            fig_pie = px.pie(
                category_data,
                values='Amount',
                names='Category',
                title="Expense Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No expenses found to categorize")
    
    @staticmethod
    def render_monthly_bar_chart(monthly_data: pd.DataFrame):
        """
        Render monthly income vs expenses bar chart.
        
        Args:
            monthly_data: DataFrame with monthly summary
        """
        st.subheader("📊 Monthly Income vs Expenses")
        
        if not monthly_data.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=monthly_data['Income'],
                name='Income',
                marker_color='green'
            ))
            fig_bar.add_trace(go.Bar(
                x=monthly_data['Month'],
                y=monthly_data['Expenses'],
                name='Expenses',
                marker_color='red'
            ))
            fig_bar.update_layout(
                title="Income vs Expenses by Month",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Insufficient data for monthly chart")
    
    @staticmethod
    def render_cumulative_line_chart(cumulative_data: pd.DataFrame):
        """
        Render cumulative balance line chart.
        
        Args:
            cumulative_data: DataFrame with cumulative balance data
        """
        st.subheader("📈 Cumulative Balance")
        
        fig_line = px.line(
            cumulative_data,
            x='Date',
            y='Cumulative_Balance',
            title="Cumulative Balance Evolution",
            labels={'Cumulative_Balance': 'Balance ($)', 'Date': 'Date'}
        )
        fig_line.update_traces(line_color='blue', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)
    
    @staticmethod
    def render_transaction_filter(categories: list):
        """
        Render transaction category filter.
        
        Args:
            categories: List of available categories
            
        Returns:
            Selected category
        """
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_category = st.selectbox(
                "Filter by category:",
                categories
            )
        return selected_category
    
    @staticmethod
    def render_transaction_table(df: pd.DataFrame, title: str = "📋 Transactions Table"):
        """
        Render transactions table.
        
        Args:
            df: DataFrame with transactions
            title: Table title
        """
        st.subheader(title)
        
        st.dataframe(
            df.sort_values('Date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def render_pl_summary_table(metrics: Dict[str, Any]):
        """
        Render P&L summary table.
        
        Args:
            metrics: Dictionary containing financial metrics
        """
        st.subheader("📊 Financial Summary")
        
        summary_data = {
            'Description': ['Total Income', 'Total Expenses', 'Net Profit/Loss'],
            'Amount ($)': [
                f'{metrics["total_income"]:,.2f}',
                f'{metrics["total_expenses"]:,.2f}',
                f'{metrics["net_balance"]:,.2f}'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
    
    @staticmethod
    def render_category_expense_table(category_data: pd.DataFrame):
        """
        Render category expense table.
        
        Args:
            category_data: DataFrame with category summary
        """
        st.subheader("📋 Expenses by Category")
        
        if not category_data.empty:
            category_display = category_data.copy()
            category_display['Amount'] = category_display['Amount'].apply(lambda x: f'$ {x:,.2f}')
            st.table(category_display.rename(columns={'Category': 'Category', 'Amount': 'Amount'}))
        else:
            st.info("No categorized expenses found")
    
    @staticmethod
    def render_pdf_generation_section():
        """
        Render PDF generation section.
        
        Returns:
            True if generate button was clicked
        """
        st.subheader("🔧 Actions")
        
        generate_clicked = st.button("📄 Generate PDF Statement", type="primary")
        
        st.markdown("---")
        st.markdown("""
        **📋 Report Information:**
        - Period covered by the data
        - Income and expense summary
        - Category breakdown
        - Generation date/time
        """)
        
        return generate_clicked
    
    @staticmethod
    def render_pdf_download_button(file_path: str, filename: str = "p&l_statement.pdf"):
        """
        Render PDF download button.
        
        Args:
            file_path: Path to the PDF file
            filename: Download filename
        """
        if os.path.exists(file_path):
            with open(file_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime='application/octet-stream'
            )
    
    @staticmethod
    def render_dashboard_charts(chart_data: Dict[str, Any]):
        """
        Render all dashboard charts in a two-column layout.
        
        Args:
            chart_data: Dictionary containing chart data
        """
        col1, col2 = st.columns(2)
        
        with col1:
            LedgerViews.render_category_pie_chart(chart_data['category_summary'])
        
        with col2:
            LedgerViews.render_monthly_bar_chart(chart_data['monthly_summary'])
        
        # Full width cumulative chart
        LedgerViews.render_cumulative_line_chart(chart_data['cumulative_data'])
    
    @staticmethod
    def render_no_data_warning(page_context: str = ""):
        """
        Render warning when no data is available.
        
        Args:
            page_context: Context for the warning message
        """
        context_msg = f" for {page_context}" if page_context else ""
        LedgerViews.render_warning_message(
            f"Please upload a file on the 'Data Upload' page first{context_msg}."
        )