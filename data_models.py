# data_models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import pandas as pd

class TransactionCategory(Enum):
    """Enumeration of supported transaction categories"""
    FOOD = "Food"
    TRANSPORTATION = "Transportation"
    INCOME = "Income"
    HEALTHCARE = "Healthcare"
    UTILITIES = "Utilities"
    ENTERTAINMENT = "Entertainment"
    HOUSING = "Housing"
    SHOPPING = "Shopping"
    OTHER = "Other"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [category.value for category in cls]
    
    @classmethod
    def is_valid_category(cls, category: str) -> bool:
        return category in cls.get_all_categories()

@dataclass
class Transaction:
    """Data model for a financial transaction"""
    date: datetime
    description: str
    amount: float
    category: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate transaction data after initialization"""
        if not isinstance(self.date, datetime):
            raise ValueError("Date must be a datetime object")
        
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Description must be a non-empty string")
        
        if not isinstance(self.amount, (int, float)):
            raise ValueError("Amount must be a number")
        
        if self.category and not TransactionCategory.is_valid_category(self.category):
            raise ValueError(f"Invalid category: {self.category}")
        
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def is_income(self) -> bool:
        return self.amount > 0
    
    @property
    def is_expense(self) -> bool:
        return self.amount < 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'Date': self.date,
            'Description': self.description,
            'Amount': self.amount,
            'Category': self.category,
            'Confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary"""
        return cls(
            date=pd.to_datetime(data['Date']),
            description=str(data['Description']),
            amount=float(data['Amount']),
            category=data.get('Category'),
            confidence=data.get('Confidence')
        )

@dataclass
class FinancialSummary:
    """Summary of financial transactions"""
    total_income: float = 0.0
    total_expenses: float = 0.0
    net_balance: float = 0.0
    transaction_count: int = 0
    category_breakdown: Dict[str, float] = field(default_factory=dict)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    @classmethod
    def from_transactions(cls, transactions: List[Transaction]) -> 'FinancialSummary':
        """Create summary from list of transactions"""
        if not transactions:
            return cls()
        
        total_income = sum(t.amount for t in transactions if t.is_income)
        total_expenses = abs(sum(t.amount for t in transactions if t.is_expense))
        net_balance = total_income - total_expenses
        
        # Category breakdown (expenses only)
        category_breakdown = {}
        for transaction in transactions:
            if transaction.is_expense and transaction.category:
                if transaction.category not in category_breakdown:
                    category_breakdown[transaction.category] = 0
                category_breakdown[transaction.category] += abs(transaction.amount)
        
        dates = [t.date for t in transactions]
        period_start = min(dates) if dates else None
        period_end = max(dates) if dates else None
        
        return cls(
            total_income=total_income,
            total_expenses=total_expenses,
            net_balance=net_balance,
            transaction_count=len(transactions),
            category_breakdown=category_breakdown,
            period_start=period_start,
            period_end=period_end
        )

@dataclass
class ModelMetrics:
    """ML Model performance metrics"""
    accuracy: float
    f1_score: float
    precision: float = 0.0
    recall: float = 0.0
    cross_val_mean: float = 0.0
    cross_val_std: float = 0.0
    training_samples: int = 0
    test_samples: int = 0
    categories_count: int = 0
    
    def is_good_performance(self) -> bool:
        """Check if model performance is acceptable"""
        return (self.accuracy >= 0.8 and 
                self.f1_score >= 0.7 and 
                self.training_samples >= 50)

# settings.py
import os
from pathlib import Path
from typing import Dict, Any
import json

class Settings:
    """Centralized application settings"""
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Data files
    TRAINING_DATA_FILE = DATA_DIR / "training_data.csv"
    SAMPLE_DATA_FILE = DATA_DIR / "mock_transactions.csv"
    
    # Model files
    MODEL_FILE = MODELS_DIR / "model.pkl"
    VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"
    MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
    
    # ML Configuration
    ML_CONFIG = {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'min_samples_per_category': 3,
        'max_features': 1000,
        'confidence_threshold': 0.5
    }
    
    # Preprocessing configuration
    PREPROCESSING_CONFIG = {
        'remove_stop_words': False,
        'min_token_length': 2,
        'max_token_length': 50,
        'preserve_amounts': True,
        'preserve_dates': True
    }
    
    # UI Configuration
    UI_CONFIG = {
        'app_title': 'Ledger Automator',
        'app_icon': '📊',
        'theme': 'light',
        'show_confidence_scores': True,
        'default_currency': 'USD',
        'date_format': '%Y-%m-%d'
    }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_config_from_file(cls, config_file: str = "config.json") -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_path = cls.PROJECT_ROOT / config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    @classmethod
    def save_config_to_file(cls, config: Dict[str, Any], config_file: str = "config.json"):
        """Save configuration to JSON file"""
        config_path = cls.PROJECT_ROOT / config_file
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def get_model_metadata(cls) -> Dict[str, Any]:
        """Get model metadata if available"""
        if cls.MODEL_METADATA_FILE.exists():
            with open(cls.MODEL_METADATA_FILE, 'r') as f:
                return json.load(f)
        
        return {}
    
    @classmethod
    def save_model_metadata(cls, metadata: Dict[str, Any]):
        """Save model metadata"""
        cls.ensure_directories()
        
        with open(cls.MODEL_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

# validators.py
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime
import re

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV file structure"""
        errors = []
        required_columns = ['Date', 'Description', 'Amount']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check minimum rows
        if len(df) == 0:
            errors.append("File is empty")
        elif len(df) < 5:
            errors.append("File has very few transactions (less than 5)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_transaction_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean transaction data"""
        warnings = []
        df_clean = df.copy()
        
        # Validate dates
        date_errors = 0
        try:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            date_errors = df_clean['Date'].isna().sum()
            if date_errors > 0:
                warnings.append(f"Found {date_errors} invalid dates")
        except Exception:
            warnings.append("Date column format is invalid")
        
        # Validate amounts
        amount_errors = 0
        try:
            df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
            amount_errors = df_clean['Amount'].isna().sum()
            if amount_errors > 0:
                warnings.append(f"Found {amount_errors} invalid amounts")
        except Exception:
            warnings.append("Amount column format is invalid")
        
        # Validate descriptions
        empty_descriptions = df_clean['Description'].isna().sum()
        if empty_descriptions > 0:
            warnings.append(f"Found {empty_descriptions} empty descriptions")
            df_clean['Description'] = df_clean['Description'].fillna('Unknown Transaction')
        
        # Remove rows with critical missing data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['Date', 'Amount'])
        final_count = len(df_clean)
        
        if initial_count != final_count:
            warnings.append(f"Removed {initial_count - final_count} rows with missing critical data")
        
        return df_clean, warnings
    
    @staticmethod
    def validate_description_quality(description: str) -> Tuple[bool, Optional[str]]:
        """Validate individual transaction description quality"""
        if not description or len(description.strip()) == 0:
            return False, "Empty description"
        
        if len(description.strip()) < 3:
            return False, "Description too short"
        
        # Check if description is mostly numbers
        if re.match(r'^[\d\s\-\.]+$', description.strip()):
            return False, "Description contains only numbers"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'^x+$',  # Only x's
            r'^\.+$', # Only dots
            r'^\?+$', # Only question marks
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, description.strip(), re.IGNORECASE):
                return False, "Suspicious description pattern"
        
        return True, None

# improved_requirements.txt
# Core ML and Data Processing
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
numpy>=1.24.0,<2.0.0
joblib>=1.3.0,<2.0.0

# Web Interface
streamlit>=1.28.0,<2.0.0
plotly>=5.15.0,<6.0.0

# PDF Generation
reportlab>=4.0.0,<5.0.0

# Testing
pytest>=7.0.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0

# Code Quality
black>=23.0.0,<24.0.0
flake8>=6.0.0,<7.0.0
mypy>=1.5.0,<2.0.0

# Development
pre-commit>=3.0.0,<4.0.0
jupyter>=1.0.0,<2.0.0

# Optional: Enhanced ML models
# transformers>=4.30.0,<5.0.0  # For BERT-based models
# torch>=2.0.0,<3.0.0          # For deep learning

# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ledger-automator",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML-powered financial transaction classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ledger-automator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "ml-enhanced": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ledger-train=scripts.train_model:main",
            "ledger-classify=scripts.classify:main",
            "ledger-app=scripts.app:main",
        ],
    },
)