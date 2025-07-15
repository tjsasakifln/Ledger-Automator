"""
Main Streamlit application using MVC (Model-View-Controller) architecture.
Separates business logic (Controller) from UI rendering (Views) for better maintainability.
"""

import streamlit as st
from controller import LedgerController
from views import LedgerViews


class LedgerApp:
    """
    Main application class that orchestrates the MVC components.
    """
    
    def __init__(self):
        """Initialize the application with controller and views."""
        self.controller = LedgerController()
        self.views = LedgerViews()
        
        # Initialize session state for controller persistence
        if 'controller' not in st.session_state:
            st.session_state.controller = self.controller
        else:
            self.controller = st.session_state.controller
    
    def run(self):
        """Main application entry point."""
        # Render page header
        self.views.render_page_header()
        
        # Render navigation and get selected page
        page = self.views.render_sidebar_navigation()
        
        # Route to appropriate page handler
        if page == "Data Upload":
            self._handle_upload_page()
        elif page == "Dashboard":
            self._handle_dashboard_page()
        elif page == "P&L Statement":
            self._handle_pl_statement_page()
    
    def _handle_upload_page(self):
        """Handle the data upload page."""
        st.header("📂 CSV File Upload")
        
        # Render file upload section
        uploaded_file, load_example = self.views.render_file_upload_section()
        
        # Handle file upload
        if uploaded_file is not None:
            result = self.controller.load_transaction_file(uploaded_file)
            
            if result['success']:
                self.views.render_success_message(result['message'])
                self.views.render_data_preview(result['data'])
                
                # Render classification button
                if self.views.render_classification_button():
                    with self.views.render_loading_spinner("Classifying transactions..."):
                        classification_result = self.controller.classify_current_transactions()
                        
                        if classification_result['success']:
                            self.views.render_success_message(classification_result['message'])
                            self.views.render_data_preview(
                                classification_result['data'], 
                                "Classified transactions preview:"
                            )
                        else:
                            self.views.render_error_message(classification_result['message'])
            else:
                self.views.render_error_message(result['message'])
        
        # Handle example data loading
        if load_example:
            result = self.controller.load_example_data()
            
            if result['success']:
                self.views.render_success_message(result['message'])
            else:
                self.views.render_error_message(result['message'])
    
    def _handle_dashboard_page(self):
        """Handle the dashboard page."""
        st.header("📈 Financial Dashboard")
        
        # Check if data is available
        if not self.controller.has_classified_transactions():
            self.views.render_no_data_warning("dashboard functionality")
            return
        
        # Get financial summary
        metrics = self.controller.get_financial_summary()
        if not metrics['success']:
            self.views.render_error_message(metrics['message'])
            return
        
        # Render financial metrics
        self.views.render_financial_metrics(metrics)
        
        st.divider()
        
        # Get chart data
        chart_data = self.controller.get_chart_data()
        if not chart_data['success']:
            self.views.render_error_message(chart_data['message'])
            return
        
        # Render dashboard charts
        self.views.render_dashboard_charts(chart_data)
        
        # Render transaction filter and table
        categories = self.controller.get_available_categories()
        selected_category = self.views.render_transaction_filter(categories)
        
        # Get filtered transactions
        filtered_df = self.controller.filter_transactions_by_category(selected_category)
        self.views.render_transaction_table(filtered_df)
    
    def _handle_pl_statement_page(self):
        """Handle the P&L statement page."""
        st.header("📄 Profit & Loss Statement")
        
        # Check if data is available
        if not self.controller.has_classified_transactions():
            self.views.render_no_data_warning("P&L statement generation")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get and render financial summary
            metrics = self.controller.get_financial_summary()
            if metrics['success']:
                self.views.render_pl_summary_table(metrics)
            else:
                self.views.render_error_message(metrics['message'])
                return
            
            # Get and render category expenses
            chart_data = self.controller.get_chart_data()
            if chart_data['success']:
                self.views.render_category_expense_table(chart_data['category_summary'])
            else:
                self.views.render_error_message(chart_data['message'])
        
        with col2:
            # Render PDF generation section
            if self.views.render_pdf_generation_section():
                with self.views.render_loading_spinner("Generating PDF..."):
                    pdf_result = self.controller.generate_pdf_statement("sample_p&l_statement.pdf")
                    
                    if pdf_result['success']:
                        self.views.render_success_message(pdf_result['message'])
                        st.info(f"📁 File saved at: {pdf_result['file_path']}")
                        
                        # Render download button
                        self.views.render_pdf_download_button(
                            pdf_result['file_path'], 
                            "p&l_statement.pdf"
                        )
                    else:
                        self.views.render_error_message(pdf_result['message'])


def main():
    """Application entry point."""
    app = LedgerApp()
    app.run()


if __name__ == "__main__":
    main()