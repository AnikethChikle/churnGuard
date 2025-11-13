"""
ChurnGuard AI - Customer Churn Prediction System
Prediction Module

This module provides functionality to make predictions on new customer data
using the trained churn prediction model.

Author: Chikle Aniketh
Version: 1.0.0
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ChurnPredictor:
    """
    Main prediction class for ChurnGuard AI system.
    
    This class handles loading trained models and making predictions
    on new customer data with probability scores and risk levels.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the ChurnPredictor.
        
        Args:
            model_dir: Directory containing saved model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metadata = None
        
        # Load all artifacts
        self.load_models()
    
    def load_models(self):
        """Load all trained model artifacts from disk."""
        try:
            print("🔄 Loading model artifacts...")
            
            # Load main model
            model_path = os.path.join(self.model_dir, 'churn_model.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ Loaded model: {model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded scaler: {scaler_path}")
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                print(f"✅ Loaded encoders: {encoders_path}")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                print(f"✅ Loaded feature names: {features_path}")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata: {metadata_path}")
            
            print("\n✅ All model artifacts loaded successfully!")
            print(f"📊 Model Type: {self.metadata.get('model_name', 'Unknown')}")
            print(f"📈 Model Accuracy: {self.metadata.get('accuracy', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def preprocess_input(self, data: Dict) -> pd.DataFrame:
        """
        Preprocess input data to match training format.
        
        Args:
            data: Dictionary containing customer information
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all expected features are present
        expected_features = [
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 'No' if feature not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'] else 0
        
        # Handle TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Encode categorical variables
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        # Handle unseen categories
                        df[col] = 0
        
        # Scale numerical features
        if self.scaler:
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Ensure correct feature order
        if self.feature_names:
            df = df[self.feature_names]
        
        return df
    
    def determine_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on churn probability.
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Risk level as string
        """
        if probability >= 0.7:
            return "🔴 HIGH RISK"
        elif probability >= 0.4:
            return "🟡 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"
    
    def predict(self, customer_data: Dict) -> Dict:
        """
        Make churn prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer information
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            X = self.preprocess_input(customer_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get probability
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                churn_probability = probabilities[1]  # Probability of churn
            else:
                churn_probability = prediction
            
            # Determine risk level
            risk_level = self.determine_risk_level(churn_probability)
            
            # Create result dictionary
            result = {
                'prediction': 'Churn' if prediction == 1 else 'No Churn',
                'churn_probability': float(churn_probability),
                'retention_probability': float(1 - churn_probability),
                'risk_level': risk_level,
                'confidence': float(max(churn_probability, 1 - churn_probability))
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, customers_data: List[Dict]) -> pd.DataFrame:
        """
        Make predictions for multiple customers.
        
        Args:
            customers_data: List of customer data dictionaries
            
        Returns:
            DataFrame with predictions for all customers
        """
        results = []
        
        print(f"\n🔄 Processing {len(customers_data)} customers...")
        
        for idx, customer in enumerate(customers_data, 1):
            try:
                prediction = self.predict(customer)
                prediction['customer_id'] = customer.get('customerID', f'Customer_{idx}')
                results.append(prediction)
                
                if idx % 100 == 0:
                    print(f"   Processed {idx}/{len(customers_data)} customers")
                    
            except Exception as e:
                print(f"⚠️ Error processing customer {idx}: {str(e)}")
                continue
        
        print(f"✅ Completed predictions for {len(results)} customers")
        
        return pd.DataFrame(results)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.metadata:
            return self.metadata
        return {
            'model_name': 'Unknown',
            'accuracy': 'N/A',
            'f1_score': 'N/A'
        }


def display_prediction_result(result: Dict):
    """
    Display prediction result in a formatted way.
    
    Args:
        result: Prediction result dictionary
    """
    print("\n" + "="*60)
    print("🛡️  CHURNGUARD AI - PREDICTION RESULT")
    print("="*60)
    print(f"\n📊 Prediction: {result['prediction']}")
    print(f"📈 Churn Probability: {result['churn_probability']:.2%}")
    print(f"🎯 Retention Probability: {result['retention_probability']:.2%}")
    print(f"⚠️  Risk Level: {result['risk_level']}")
    print(f"💪 Confidence: {result['confidence']:.2%}")
    print("\n" + "="*60)


def main():
    """Main function to demonstrate prediction usage."""
    
    print("="*60)
    print("🛡️  CHURNGUARD AI - PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Sample customer data for prediction
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 80.0,
        'TotalCharges': 960.0
    }
    
    print("\n📋 Sample Customer Profile:")
    print("-" * 60)
    for key, value in sample_customer.items():
        print(f"   {key:20s}: {value}")
    
    # Make prediction
    print("\n🔮 Making prediction...")
    result = predictor.predict(sample_customer)
    
    # Display result
    display_prediction_result(result)
    
    # Show model info
    print("\n📊 Model Information:")
    print("-" * 60)
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"   {key:20s}: {value}")
    
    print("\n✅ Prediction complete!")
    print("\n💡 Usage Example:")
    print("""
    from predict import ChurnPredictor
    
    predictor = ChurnPredictor()
    result = predictor.predict(customer_data)
    print(f"Churn Risk: {result['churn_probability']:.2%}")
    """)


if __name__ == "__main__":
    main()