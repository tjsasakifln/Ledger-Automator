# tests/conftest.py
"""
PyTest configuration and shared fixtures for Ledger Automator tests
Provides common test utilities, fixtures, and configuration
"""

import pytest
import tempfile
import pandas as pd
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path
import shutil
from datetime import datetime
import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest settings"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def sample_csv_data():
    """Sample CSV data for testing"""
    return """Date,Description,Amount
2024-01-01,Grocery Store Purchase,-150.50
2024-01-02,Gas Station Fill-up,-80.00
2024-01-03,Salary Payment,3500.00
2024-01-04,Restaurant Dinner,-45.25
2024-01-05,Online Shopping,-125.00
2024-01-06,ATM Withdrawal,-100.00
2024-01-07,Freelance Payment,750.00
2024-01-08,Utility Bill Payment,-85.50
2024-01-09,Coffee Shop,-6.50
2024-01-10,Investment Dividend,200.00"""

@pytest.fixture
def sample_dataframe(sample_csv_data):
    """Sample DataFrame for testing"""
    from io import StringIO
    return pd.read_csv(StringIO(sample_csv_data))

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit session state and functions"""
    with patch.dict('streamlit.session_state', {}, clear=True):
        # Mock common Streamlit functions
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.success') as mock_success:
            
            yield {
                'error': mock_error,
                'warning': mock_warning,
                'info': mock_info,
                'success': mock_success,
                'session_state': st.session_state
            }

@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    transactions = []
    descriptions = [
        'Supermarket ABC', 'Gas Station XYZ', 'Salary Payment',
        'Netflix Subscription', 'Pharmacy Purchase', 'Restaurant Dinner',
        'Uber Ride', 'Electricity Bill', 'Grocery Store', 'Coffee Shop'
    ]
    
    categories = [
        'Food', 'Transportation', 'Income', 'Entertainment', 'Healthcare',
        'Food', 'Transportation', 'Utilities', 'Food', 'Food'
    ]
    
    amounts = [
        -150.50, -45.20, 3500.00, -15.99, -25.30,
        -85.75, -18.50, -120.00, -200.25, -12.50
    ]
    
    for i in range(len(descriptions)):
        transactions.append({
            'Date': dates[i % len(dates)],
            'Description': descriptions[i],
            'Amount': amounts[i],
            'Category': categories[i]
        })
    
    return pd.DataFrame(transactions)

@pytest.fixture
def sample_training_data():
    """Sample training data with more examples"""
    data = []
    
    # Food transactions
    food_descriptions = [
        'Supermarket Extra', 'Grocery Store ABC', 'Restaurant XYZ',
        'McDonald\'s', 'Starbucks Coffee', 'Pizza Hut', 'Local Market'
    ]
    
    # Transportation
    transport_descriptions = [
        'Gas Station Shell', 'Uber Technologies', 'Taxi Ride',
        'Bus Ticket', 'Metro Card', 'Parking Fee'
    ]
    
    # Add food examples
    for desc in food_descriptions:
        data.append({'Description': desc, 'Category': 'Food'})
    
    # Add transportation examples
    for desc in transport_descriptions:
        data.append({'Description': desc, 'Category': 'Transportation'})
    
    # Add other categories
    other_examples = [
        ('Netflix Streaming', 'Entertainment'),
        ('Spotify Premium', 'Entertainment'),
        ('Salary Company', 'Income'),
        ('Freelance Payment', 'Income'),
        ('Electric Bill', 'Utilities'),
        ('Internet Provider', 'Utilities'),
        ('Doctor Visit', 'Healthcare'),
        ('Pharmacy CVS', 'Healthcare'),
        ('Rent Payment', 'Housing'),
        ('Mortgage Bank', 'Housing')
    ]
    
    for desc, cat in other_examples:
        data.append({'Description': desc, 'Category': cat})
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(sample_transactions):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        sample_transactions.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# tests/test_preprocess.py
import pytest
import pandas as pd
import numpy as np
from scripts.preprocess import (
    clean_text, preprocess_descriptions, create_vectorizer,
    vectorize_text, prepare_training_data, prepare_prediction_data
)

class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        # Test normal text
        result = clean_text("Supermarket ABC 123")
        assert "supermarket abc" in result.lower()
        
        # Test with special characters
        result = clean_text("Restaurant-XYZ!@#$%")
        assert result == "restaurant xyz"
        
        # Test with None/NaN
        result = clean_text(None)
        assert result == ""
        
        result = clean_text(pd.NA)
        assert result == ""
    
    def test_clean_text_preserves_important_info(self):
        """Test that cleaning preserves important information"""
        # Test that it doesn't completely remove everything
        result = clean_text("ATM Withdrawal $50.00")
        assert len(result) > 0
        assert "atm" in result
        assert "withdrawal" in result
    
    def test_preprocess_descriptions(self):
        """Test preprocessing of description lists"""
        descriptions = [
            "Supermarket ABC",
            "Gas Station XYZ", 
            None,
            "Restaurant-123!"
        ]
        
        result = preprocess_descriptions(descriptions)
        
        assert len(result) == 4
        assert all(isinstance(desc, str) for desc in result)
        assert result[2] == ""  # None should become empty string
    
    def test_create_vectorizer(self):
        """Test vectorizer creation"""
        vectorizer = create_vectorizer(max_features=100)
        
        assert hasattr(vectorizer, 'fit_transform')
        assert vectorizer.max_features == 100
        assert vectorizer.lowercase == True
    
    def test_vectorize_text(self):
        """Test text vectorization"""
        descriptions = [
            "supermarket grocery food",
            "gas station fuel car",
            "restaurant dinner food"
        ]
        
        X_vectorized, vectorizer = vectorize_text(descriptions, fit=True)
        
        assert X_vectorized.shape[0] == 3  # 3 documents
        assert X_vectorized.shape[1] > 0   # Some features
        assert vectorizer is not None
    
    def test_prepare_training_data(self, sample_training_data):
        """Test training data preparation"""
        X, y, vectorizer = prepare_training_data(sample_training_data)
        
        assert X.shape[0] == len(sample_training_data)
        assert len(y) == len(sample_training_data)
        assert vectorizer is not None
        
        # Check that categories are preserved
        unique_categories = set(y)
        expected_categories = set(sample_training_data['Category'].unique())
        assert unique_categories == expected_categories
    
    def test_prepare_prediction_data(self, sample_transactions, sample_training_data):
        """Test prediction data preparation"""
        # First prepare training data to get vectorizer
        _, _, vectorizer = prepare_training_data(sample_training_data)
        
        # Then prepare prediction data
        X_pred = prepare_prediction_data(sample_transactions, vectorizer)
        
        assert X_pred.shape[0] == len(sample_transactions)
        assert X_pred.shape[1] == vectorizer.vocabulary_.__len__()

# tests/test_utils.py
import pytest
import pandas as pd
import os
from scripts.utils import (
    load_data, validate_transaction_data, save_classified_data,
    create_summary_stats, format_currency, ensure_directory_exists
)

class TestUtils:
    """Test utility functions"""
    
    def test_load_data(self, temp_csv_file):
        """Test data loading from CSV"""
        df = load_data(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Date' in df.columns
        assert 'Description' in df.columns
        assert 'Amount' in df.columns
    
    def test_load_data_with_required_columns(self, temp_csv_file):
        """Test data loading with column validation"""
        required_columns = ['Date', 'Description', 'Amount']
        df = load_data(temp_csv_file, required_columns)
        
        assert all(col in df.columns for col in required_columns)
    
    def test_load_data_missing_file(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_data('non_existent_file.csv')
    
    def test_validate_transaction_data(self, sample_transactions):
        """Test transaction data validation"""
        df_validated = validate_transaction_data(sample_transactions)
        
        assert isinstance(df_validated, pd.DataFrame)
        assert len(df_validated) <= len(sample_transactions)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df_validated['Date'])
        assert pd.api.types.is_numeric_dtype(df_validated['Amount'])
    
    def test_validate_transaction_data_with_bad_data(self):
        """Test validation with problematic data"""
        bad_data = pd.DataFrame({
            'Date': ['2024-01-01', 'invalid_date', '2024-01-03'],
            'Description': ['Valid', 'Also Valid', 'Valid Too'],
            'Amount': [100.0, 'not_a_number', 200.0]
        })
        
        df_validated = validate_transaction_data(bad_data)
        
        # Should remove rows with invalid data
        assert len(df_validated) < len(bad_data)
        assert df_validated['Amount'].notna().all()
    
    def test_save_classified_data(self, sample_transactions, temp_model_dir):
        """Test saving classified data"""
        output_path = os.path.join(temp_model_dir, 'test_output.csv')
        
        result_path = save_classified_data(
            sample_transactions, 
            output_path, 
            include_timestamp=False
        )
        
        assert os.path.exists(result_path)
        
        # Verify saved data can be loaded
        loaded_df = pd.read_csv(result_path)
        assert len(loaded_df) == len(sample_transactions)
    
    def test_create_summary_stats(self, sample_transactions):
        """Test summary statistics calculation"""
        stats = create_summary_stats(sample_transactions)
        
        assert 'financeiro' in stats
        assert 'total_receitas' in stats['financeiro']
        assert 'total_despesas' in stats['financeiro']
        assert 'saldo_liquido' in stats['financeiro']
        
        # Verify calculations
        income = sample_transactions[sample_transactions['Amount'] > 0]['Amount'].sum()
        assert stats['financeiro']['total_receitas'] == income
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1234.56) == "R$ 1.234,56"
        assert format_currency(0) == "R$ 0,00"
        assert format_currency(-100.50) == "R$ -100,50"
    
    def test_ensure_directory_exists(self, temp_model_dir):
        """Test directory creation"""
        new_dir = os.path.join(temp_model_dir, 'new_directory')
        
        assert not os.path.exists(new_dir)
        ensure_directory_exists(new_dir)
        assert os.path.exists(new_dir)
        
        # Should not raise error if directory already exists
        ensure_directory_exists(new_dir)

# tests/test_ml_pipeline.py
import pytest
import pandas as pd
import numpy as np
from scripts.ml_pipeline import ModelTrainer

class TestMLPipeline:
    """Test ML pipeline functionality"""
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(random_state=42)
        
        assert trainer.random_state == 42
        assert trainer.best_model is None
        assert trainer.best_vectorizer is None
        assert len(trainer.models_to_try) > 0
    
    def test_validate_training_data(self, sample_training_data):
        """Test training data validation"""
        trainer = ModelTrainer()
        
        validated_df = trainer.validate_training_data(sample_training_data)
        
        assert isinstance(validated_df, pd.DataFrame)
        assert len(validated_df) <= len(sample_training_data)
        assert 'Description' in validated_df.columns
        assert 'Category' in validated_df.columns
    
    def test_validate_training_data_insufficient(self):
        """Test validation with insufficient data"""
        trainer = ModelTrainer()
        
        # Very small dataset
        small_data = pd.DataFrame({
            'Description': ['test1', 'test2'],
            'Category': ['cat1', 'cat2']
        })
        
        # Should not raise error but should log warnings
        validated_df = trainer.validate_training_data(small_data)
        assert len(validated_df) == 2
    
    def test_prepare_data_with_validation(self, sample_training_data):
        """Test data preparation with validation"""
        trainer = ModelTrainer()
        
        X, y, vectorizer, class_weights = trainer.prepare_data_with_validation(sample_training_data)
        
        assert X.shape[0] == len(sample_training_data)
        assert len(y) == len(sample_training_data)
        assert vectorizer is not None
        assert isinstance(class_weights, dict)
        assert len(class_weights) > 0
    
    def test_train_with_cross_validation(self, sample_training_data):
        """Test cross-validation training"""
        trainer = ModelTrainer()
        
        X, y, vectorizer, class_weights = trainer.prepare_data_with_validation(sample_training_data)
        
        best_model_name, model_scores = trainer.train_with_cross_validation(
            X, y, vectorizer, class_weights
        )
        
        assert best_model_name in trainer.models_to_try.keys()
        assert isinstance(model_scores, dict)
        assert len(model_scores) > 0
        
        # Check that scores are reasonable
        for model_name, scores in model_scores.items():
            assert 'mean_score' in scores
            assert 'std_score' in scores
            assert 0 <= scores['mean_score'] <= 1

# tests/test_integration.py
import pytest
import pandas as pd
import os
import tempfile
from scripts.ml_pipeline import ModelTrainer

class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_training_pipeline(self, sample_training_data, temp_model_dir):
        """Test the complete training pipeline"""
        trainer = ModelTrainer()
        
        # Run full pipeline
        model, vectorizer, evaluation_results, model_scores = trainer.train_full_pipeline(
            sample_training_data
        )
        
        assert model is not None
        assert vectorizer is not None
        assert isinstance(evaluation_results, dict)
        assert 'accuracy' in evaluation_results
        assert isinstance(model_scores, dict)
        
        # Test saving
        model_path, vectorizer_path = trainer.save_model(temp_model_dir)
        
        assert os.path.exists(model_path)
        assert os.path.exists(vectorizer_path)
    
    def test_classification_pipeline(self, sample_training_data, sample_transactions, temp_model_dir):
        """Test training followed by classification"""
        # 1. Train model
        trainer = ModelTrainer()
        model, vectorizer, _, _ = trainer.train_full_pipeline(sample_training_data)
        trainer.save_model(temp_model_dir)
        
        # 2. Load and classify
        import joblib
        from scripts.preprocess import prepare_prediction_data
        
        loaded_model = joblib.load(os.path.join(temp_model_dir, 'model.pkl'))
        loaded_vectorizer = joblib.load(os.path.join(temp_model_dir, 'vectorizer.pkl'))
        
        # Prepare prediction data
        X_pred = prepare_prediction_data(sample_transactions, loaded_vectorizer)
        
        # Make predictions
        predictions = loaded_model.predict(X_pred)
        
        assert len(predictions) == len(sample_transactions)
        assert all(isinstance(pred, str) for pred in predictions)

# pytest configuration
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes

# Run with: pytest tests/ -v