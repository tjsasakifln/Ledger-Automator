"""
Data augmentation module for expanding training dataset.
Creates synthetic training examples to improve model robustness.
"""

import pandas as pd
import random
import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataAugmenter:
    """Generates augmented financial transaction data for training."""
    
    def __init__(self):
        self.merchant_variations = {
            'supermarket': [
                'Supermarket', 'Super Market', 'Grocery Store', 'Market', 'Food Store',
                'Whole Foods', 'Safeway', 'Kroger', 'Target', 'Walmart Grocery'
            ],
            'gas_station': [
                'Shell', 'BP', 'Exxon', 'Chevron', 'Mobil', 'Gas Station', 'Fuel',
                'Petrol Station', 'Service Station', 'Quick Stop'
            ],
            'restaurant': [
                'McDonald\'s', 'Burger King', 'KFC', 'Subway', 'Pizza Hut',
                'Restaurant', 'Cafe', 'Bistro', 'Diner', 'Fast Food'
            ],
            'pharmacy': [
                'CVS', 'Walgreens', 'Rite Aid', 'Pharmacy', 'Drugstore',
                'Medical Store', 'Health Store', 'Chemist'
            ],
            'bank': [
                'Bank of America', 'Chase', 'Wells Fargo', 'Citibank', 'Bank',
                'Credit Union', 'Financial Services', 'Banking'
            ]
        }
        
        self.location_suffixes = [
            'Downtown', 'Mall', 'Center', 'Plaza', 'Square', 'Main St',
            'Broadway', 'Central', 'North', 'South', 'East', 'West'
        ]
        
        self.financial_terms = [
            'Payment', 'Purchase', 'Transaction', 'Charge', 'Fee', 'Service',
            'Withdrawal', 'Deposit', 'Transfer', 'Credit', 'Debit'
        ]

    def generate_food_transactions(self, count: int = 50) -> List[Tuple[str, str]]:
        """Generate food-related transactions."""
        transactions = []
        food_merchants = self.merchant_variations['supermarket'] + self.merchant_variations['restaurant']
        
        for _ in range(count):
            merchant = random.choice(food_merchants)
            location = random.choice(self.location_suffixes) if random.random() > 0.3 else ""
            term = random.choice(self.financial_terms) if random.random() > 0.7 else ""
            
            description = f"{merchant}"
            if location:
                description += f" {location}"
            if term:
                description = f"{term} {description}"
            
            transactions.append((description, "Food"))
        
        return transactions

    def generate_transportation_transactions(self, count: int = 40) -> List[Tuple[str, str]]:
        """Generate transportation-related transactions."""
        transactions = []
        transport_merchants = self.merchant_variations['gas_station'] + [
            'Uber', 'Lyft', 'Taxi', 'Metro', 'Bus', 'Train', 'Parking',
            'Toll Road', 'Highway', 'Bridge Toll', 'Airport Parking'
        ]
        
        for _ in range(count):
            merchant = random.choice(transport_merchants)
            location = random.choice(self.location_suffixes) if random.random() > 0.5 else ""
            
            description = f"{merchant}"
            if location and merchant not in ['Uber', 'Lyft']:
                description += f" {location}"
            
            transactions.append((description, "Transportation"))
        
        return transactions

    def generate_income_transactions(self, count: int = 30) -> List[Tuple[str, str]]:
        """Generate income-related transactions."""
        transactions = []
        income_sources = [
            'Salary', 'Payroll', 'Wage', 'Freelance Payment', 'Consulting Fee',
            'Investment Return', 'Dividend', 'Interest', 'Bonus', 'Commission',
            'Contract Payment', 'Project Fee', 'Refund', 'Tax Refund',
            'Insurance Claim', 'Rental Income', 'Side Income', 'Part Time Job'
        ]
        
        companies = [
            'Tech Corp', 'Global Solutions', 'Innovation Ltd', 'Digital Agency',
            'Consulting Group', 'Services Inc', 'Enterprise Co', 'Systems LLC'
        ]
        
        for _ in range(count):
            if random.random() > 0.3:
                source = random.choice(income_sources)
                company = random.choice(companies) if random.random() > 0.5 else ""
                description = f"{source}"
                if company and source in ['Salary', 'Payroll', 'Bonus']:
                    description += f" {company}"
            else:
                description = random.choice(income_sources)
            
            transactions.append((description, "Income"))
        
        return transactions

    def generate_healthcare_transactions(self, count: int = 35) -> List[Tuple[str, str]]:
        """Generate healthcare-related transactions."""
        transactions = []
        healthcare_merchants = self.merchant_variations['pharmacy'] + [
            'Doctor', 'Dentist', 'Hospital', 'Clinic', 'Medical Center',
            'Urgent Care', 'Emergency Room', 'Specialist', 'Therapist',
            'Gym', 'Fitness Center', 'Yoga Studio', 'Health Club',
            'Laboratory', 'Radiology', 'Physical Therapy'
        ]
        
        for _ in range(count):
            merchant = random.choice(healthcare_merchants)
            location = random.choice(self.location_suffixes) if random.random() > 0.4 else ""
            
            description = f"{merchant}"
            if location:
                description += f" {location}"
            
            transactions.append((description, "Healthcare"))
        
        return transactions

    def generate_utilities_transactions(self, count: int = 25) -> List[Tuple[str, str]]:
        """Generate utilities-related transactions."""
        transactions = []
        utility_companies = [
            'Electric Company', 'Power & Light', 'Gas Company', 'Water Department',
            'Internet Provider', 'Cable TV', 'Phone Service', 'Mobile Carrier',
            'Verizon', 'AT&T', 'Comcast', 'Spectrum', 'T-Mobile', 'Sprint'
        ]
        
        service_types = [
            'Electric Bill', 'Gas Bill', 'Water Bill', 'Internet Bill',
            'Phone Bill', 'Cable Bill', 'Utility Payment', 'Service Fee'
        ]
        
        for _ in range(count):
            if random.random() > 0.4:
                description = random.choice(utility_companies)
            else:
                description = random.choice(service_types)
            
            transactions.append((description, "Utilities"))
        
        return transactions

    def generate_entertainment_transactions(self, count: int = 30) -> List[Tuple[str, str]]:
        """Generate entertainment-related transactions."""
        transactions = []
        entertainment_sources = [
            'Netflix', 'Spotify', 'Amazon Prime', 'Disney+', 'Hulu', 'HBO Max',
            'YouTube Premium', 'Apple Music', 'Cinema', 'Theater', 'Concert',
            'Sports Event', 'Streaming Service', 'Gaming', 'Books', 'Magazine'
        ]
        
        for _ in range(count):
            source = random.choice(entertainment_sources)
            description = f"{source}"
            if source in ['Cinema', 'Theater', 'Concert'] and random.random() > 0.5:
                location = random.choice(self.location_suffixes)
                description += f" {location}"
            
            transactions.append((description, "Entertainment"))
        
        return transactions

    def generate_housing_transactions(self, count: int = 20) -> List[Tuple[str, str]]:
        """Generate housing-related transactions."""
        transactions = []
        housing_types = [
            'Rent Payment', 'Mortgage Payment', 'Property Tax', 'HOA Fee',
            'Condo Fee', 'Property Management', 'Maintenance Fee',
            'Home Insurance', 'Property Insurance', 'Real Estate'
        ]
        
        for _ in range(count):
            description = random.choice(housing_types)
            transactions.append((description, "Housing"))
        
        return transactions

    def generate_shopping_transactions(self, count: int = 40) -> List[Tuple[str, str]]:
        """Generate shopping-related transactions."""
        transactions = []
        shopping_places = [
            'Amazon', 'eBay', 'Target', 'Walmart', 'Best Buy', 'Home Depot',
            'Costco', 'Mall', 'Department Store', 'Clothing Store', 'Electronics',
            'Online Store', 'Retail Store', 'Shopping Center', 'Outlet'
        ]
        
        for _ in range(count):
            place = random.choice(shopping_places)
            description = f"{place}"
            if place in ['Mall', 'Shopping Center'] and random.random() > 0.5:
                location = random.choice(self.location_suffixes)
                description += f" {location}"
            
            transactions.append((description, "Shopping"))
        
        return transactions

    def generate_augmented_dataset(self) -> pd.DataFrame:
        """Generate complete augmented dataset with balanced categories."""
        logger.info("Generating augmented training dataset...")
        
        all_transactions = []
        
        # Generate transactions for each category
        all_transactions.extend(self.generate_food_transactions(60))
        all_transactions.extend(self.generate_transportation_transactions(45))
        all_transactions.extend(self.generate_income_transactions(35))
        all_transactions.extend(self.generate_healthcare_transactions(40))
        all_transactions.extend(self.generate_utilities_transactions(30))
        all_transactions.extend(self.generate_entertainment_transactions(35))
        all_transactions.extend(self.generate_housing_transactions(25))
        all_transactions.extend(self.generate_shopping_transactions(45))
        
        # Create DataFrame
        df = pd.DataFrame(all_transactions, columns=['Description', 'Category'])
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} augmented training samples")
        logger.info(f"Category distribution:\n{df['Category'].value_counts()}")
        
        return df

def create_enhanced_training_data():
    """Create enhanced training dataset by combining original and augmented data."""
    # Load original data
    original_path = "data/training_data.csv"
    original_df = pd.read_csv(original_path)
    
    # Generate augmented data
    augmenter = FinancialDataAugmenter()
    augmented_df = augmenter.generate_augmented_dataset()
    
    # Combine datasets
    enhanced_df = pd.concat([original_df, augmented_df], ignore_index=True)
    enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save enhanced dataset
    enhanced_path = "data/enhanced_training_data.csv"
    enhanced_df.to_csv(enhanced_path, index=False)
    
    logger.info(f"Enhanced training dataset saved to {enhanced_path}")
    logger.info(f"Total samples: {len(enhanced_df)}")
    logger.info(f"Final category distribution:\n{enhanced_df['Category'].value_counts()}")
    
    return enhanced_df

if __name__ == "__main__":
    # Generate enhanced training data
    create_enhanced_training_data()