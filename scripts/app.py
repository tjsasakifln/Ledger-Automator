import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_csv, 
    classify_transactions, 
    calculate_monthly_summary,
    calculate_category_summary,
    calculate_cumulative_balance,
    generate_pl_statement
)
import os

st.set_page_config(
    page_title="Ledger Automator",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Ledger Automator")
    st.markdown("Sistema de classificação automática de transações financeiras")
    
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Escolha uma opção:",
        ["Upload de Dados", "Dashboard", "Demonstrativo P&L"]
    )
    
    if page == "Upload de Dados":
        upload_page()
    elif page == "Dashboard":
        dashboard_page()
    else:
        pl_statement_page()

def upload_page():
    st.header("📂 Upload de Arquivo CSV")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Formato esperado do arquivo CSV:**
        - Colunas obrigatórias: `Date`, `Description`, `Amount`
        - Date: formato YYYY-MM-DD
        - Description: descrição da transação
        - Amount: valor (positivo para receitas, negativo para despesas)
        """)
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=['csv'],
            help="Arquivo deve conter colunas: Date, Description, Amount"
        )
        
        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado com sucesso! {len(df)} transações encontradas.")
                
                st.subheader("Preview dos dados:")
                st.dataframe(df.head(10))
                
                if st.button("🤖 Classificar Transactions", type="primary"):
                    with st.spinner("Classificando transações..."):
                        df_classified = classify_transactions(df)
                        st.session_state['transactions'] = df_classified
                        st.success("✅ Transactions classificadas com sucesso!")
                        st.dataframe(df_classified.head(10))
                        
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    
    with col2:
        st.markdown("### 📄 Dados de Exemplo")
        if st.button("Carregar dados de exemplo"):
            try:
                example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_transactions.csv')
                df = load_csv(example_path)
                df_classified = classify_transactions(df)
                st.session_state['transactions'] = df_classified
                st.success("✅ Dados de exemplo carregados!")
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")

def dashboard_page():
    st.header("📈 Dashboard Financeiro")
    
    if 'transactions' not in st.session_state:
        st.warning("⚠️ Primeiro faça upload de um arquivo na página 'Upload de Dados'")
        return
    
    df = st.session_state['transactions']
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_receitas = df[df['Amount'] > 0]['Amount'].sum()
    total_despesas = abs(df[df['Amount'] < 0]['Amount'].sum())
    saldo_liquido = total_receitas - total_despesas
    num_transacoes = len(df)
    
    with col1:
        st.metric("💰 Total Income", f"$ {total_receitas:,.2f}")
    with col2:
        st.metric("💸 Total Expenses", f"$ {total_despesas:,.2f}")
    with col3:
        st.metric("📊 Net Balance", f"$ {saldo_liquido:,.2f}")
    with col4:
        st.metric("🔢 Transactions", num_transacoes)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍰 Expenses by Category")
        category_summary = calculate_category_summary(df)
        if not category_summary.empty:
            fig_pie = px.pie(
                category_summary,
                values='Amount',
                names='Category',
                title="Distribuição de Despesas"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Nenhuma despesa encontrada para categorizar")
    
    with col2:
        st.subheader("📊 Monthly Income vs Expenses")
        monthly_summary = calculate_monthly_summary(df)
        if not monthly_summary.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=monthly_summary['Month'],
                y=monthly_summary['Receita'],
                name='Receita',
                marker_color='green'
            ))
            fig_bar.add_trace(go.Bar(
                x=monthly_summary['Month'],
                y=monthly_summary['Despesa'],
                name='Despesa',
                marker_color='red'
            ))
            fig_bar.update_layout(
                title="Receitas vs Despesas por Mês",
                xaxis_title="Mês",
                yaxis_title="Valor ($)",
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Dados insuficientes para gráfico mensal")
    
    st.subheader("📈 Cumulative Balance")
    cumulative_data = calculate_cumulative_balance(df)
    fig_line = px.line(
        cumulative_data,
        x='Date',
        y='Cumulative_Balance',
        title="Evolução do Cumulative Balance",
        labels={'Cumulative_Balance': 'Saldo ($)', 'Date': 'Data'}
    )
    fig_line.update_traces(line_color='blue', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("📋 Tabela de Transactions")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_category = st.selectbox(
            "Filter by category:",
            ['All'] + list(df['Category'].unique())
        )
    
    filtered_df = df if selected_category == 'All' else df[df['Category'] == selected_category]
    
    st.dataframe(
        filtered_df.sort_values('Date', ascending=False),
        use_container_width=True,
        hide_index=True
    )

def pl_statement_page():
    st.header("📄 Profit & Loss Statement")
    
    if 'transactions' not in st.session_state:
        st.warning("⚠️ Primeiro faça upload de um arquivo na página 'Upload de Dados'")
        return
    
    df = st.session_state['transactions']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Financial Summary")
        
        total_receitas = df[df['Amount'] > 0]['Amount'].sum()
        total_despesas = abs(df[df['Amount'] < 0]['Amount'].sum())
        lucro_liquido = total_receitas - total_despesas
        
        summary_data = {
            'Descrição': ['Total de Receitas', 'Total de Despesas', 'Lucro/Prejuízo Líquido'],
            'Valor ($)': [f'{total_receitas:,.2f}', f'{total_despesas:,.2f}', f'{lucro_liquido:,.2f}']
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        st.subheader("📋 Expenses by Category")
        category_summary = calculate_category_summary(df)
        if not category_summary.empty:
            category_summary['Amount'] = category_summary['Amount'].apply(lambda x: f'$ {x:,.2f}')
            st.table(category_summary.rename(columns={'Category': 'Categoria', 'Amount': 'Valor'}))
        else:
            st.info("Nenhuma despesa categorizada encontrada")
    
    with col2:
        st.subheader("🔧 Ações")
        
        if st.button("📄 Generate PDF Statement", type="primary"):
            output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'sample_p&l_statement.pdf')
            
            with st.spinner("Gerando PDF..."):
                success = generate_pl_statement(df, output_path)
                
                if success:
                    st.success("✅ PDF generated successfully!")
                    st.info(f"📁 File saved at: {output_path}")
                    
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=PDFbyte,
                            file_name="demonstrativo_pl.pdf",
                            mime='application/octet-stream'
                        )
                else:
                    st.error("❌ Error generating PDF")
        
        st.markdown("---")
        st.markdown("""
        **📋 Informações do Relatório:**
        - Período coberto pelos dados
        - Resumo de receitas e despesas
        - Detalhamento por categoria
        - Data/hora de geração
        """)

if __name__ == "__main__":
    main()