"""
ChurnGuard AI - Customer Churn Prediction System
================================================
Main training script for the complete machine learning pipeline.

"""
import sys
import io

# Fix Unicode encoding for Windows console
# This prevents UnicodeEncodeError when using emojis in logging
if sys.platform == 'win32':
    # Reconfigure stdout and stderr to use UTF-8 encoding
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
import os
import sys
import warnings
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
import pickle
import json

warnings.filterwarnings('ignore')

# Configure logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class Config:
    """Configuration class for the project"""
    DATA_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
    MODEL_DIR = 'models/'
    OUTPUT_DIR = 'outputs/'
    VIZ_DIR = 'outputs/visualizations/'
    REPORT_DIR = 'outputs/reports/'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
    TARGET_COLUMN = 'Churn'
    ID_COLUMN = 'customerID'
    
    @staticmethod
    def create_directories():
        directories = [Config.MODEL_DIR, Config.OUTPUT_DIR, Config.VIZ_DIR, 
                      Config.REPORT_DIR, 'data/raw', 'data/processed']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("✅ Directories created successfully")

def load_data(filepath):
    logger.info("="*80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*80)
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filepath}")
        logger.info("\n📥 Download dataset from: https://www.kaggle.com/blastchar/telco-customer-churn")
        sys.exit(1)

def explore_data(df):
    logger.info("\n" + "="*80)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*80)
    logger.info(f"\n📊 Dataset Shape: {df.shape}")
    logger.info(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
    
    churn_counts = df['Churn'].value_counts()
    churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
    logger.info(f"\n🎯 Target Distribution:")
    logger.info(f"   No Churn: {churn_counts['No']} ({100-churn_rate:.2f}%)")
    logger.info(f"   Churn: {churn_counts['Yes']} ({churn_rate:.2f}%)")
    
    create_eda_visualizations(df)
    return churn_rate

def create_eda_visualizations(df):
    # 1. Churn Distribution
    plt.figure(figsize=(8, 6))
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    plt.bar(['No Churn', 'Churn'], churn_counts.values, color=colors, edgecolor='black')
    plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xlabel('Churn Status', fontsize=12)
    for i, v in enumerate(churn_counts.values):
        plt.text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: churn_distribution.png")
    
    # 2. Churn by Contract Type
    plt.figure(figsize=(10, 6))
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'])
    plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Contract Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(['No Churn', 'Churn'], loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}churn_by_contract.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: churn_by_contract.png")
    
    # 3. Tenure Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df[df['Churn'] == 'No']['tenure'].hist(bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
    plt.title('Tenure Distribution - No Churn', fontweight='bold')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    df[df['Churn'] == 'Yes']['tenure'].hist(bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
    plt.title('Tenure Distribution - Churn', fontweight='bold')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}tenure_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: tenure_distribution.png")

def preprocess_data(df):
    logger.info("\n" + "="*80)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("="*80)
    
    df_processed = df.copy()
    df_processed = df_processed.drop(Config.ID_COLUMN, axis=1)
    logger.info("✅ Dropped customer ID column")
    
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(0, inplace=True)
    logger.info("✅ Handled missing values")
    
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    logger.info("✅ Converted target variable to binary")
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    logger.info(f"✅ Encoded {len(categorical_cols)} categorical variables")
    
    X = df_processed.drop(Config.TARGET_COLUMN, axis=1)
    y = df_processed[Config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    logger.info(f"\n✂️ Data Split: Training={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train[Config.NUMERICAL_FEATURES] = scaler.fit_transform(X_train[Config.NUMERICAL_FEATURES])
    X_test[Config.NUMERICAL_FEATURES] = scaler.transform(X_test[Config.NUMERICAL_FEATURES])
    logger.info("✅ Scaled numerical features")
    
    df_processed.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"✅ Saved processed data")
    
    preprocessors = {'scaler': scaler, 'label_encoders': label_encoders, 'feature_names': list(X.columns)}
    return X_train, X_test, y_train, y_test, preprocessors

def train_models(X_train, y_train):
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=Config.RANDOM_STATE, max_depth=10, min_samples_split=20),
        'Random Forest': RandomForestClassifier(random_state=Config.RANDOM_STATE, n_estimators=100, max_depth=20, min_samples_split=10, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=Config.RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"\n🔄 Training {name}...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS, scoring='f1')
        training_time = (datetime.now() - start_time).total_seconds()
        trained_models[name] = model
        logger.info(f"   ✅ Completed in {training_time:.2f}s | CV F1: {cv_scores.mean():.4f}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*80)
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
        }
        results.append(metrics)
        logger.info(f"\n📊 {name}: Acc={metrics['Accuracy']:.4f} | F1={metrics['F1-Score']:.4f}")
    
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    logger.info(f"\n🏆 BEST MODEL: {best_model_name}")
    
    create_model_comparison_plot(results_df)
    return results_df, best_model_name

def create_model_comparison_plot(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric], color='steelblue', edgecolor='black', linewidth=1.5)
        best_idx = results_df[metric].idxmax()
        bars[best_idx].set_color('#e74c3c')
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, value in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: model_comparison.png")

def analyze_best_model(model, model_name, X_test, y_test, feature_names):
    logger.info("\n" + "="*80)
    logger.info(f"STEP 6: DETAILED ANALYSIS - {model_name}")
    logger.info("="*80)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n📋 Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'], cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{Config.VIZ_DIR}roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: roc_curve.png")
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        logger.info(f"\n📊 Top 10 Features:\n{feature_importance.head(10).to_string(index=False)}")
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top 15 Features - {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{Config.VIZ_DIR}feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Saved: feature_importance.png")

def save_artifacts(model, model_name, preprocessors, results_df):
    logger.info("\n" + "="*80)
    logger.info("STEP 7: SAVING MODELS AND ARTIFACTS")
    logger.info("="*80)
    
    with open(f'{Config.MODEL_DIR}churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{Config.MODEL_DIR}scaler.pkl', 'wb') as f:
        pickle.dump(preprocessors['scaler'], f)
    with open(f'{Config.MODEL_DIR}label_encoders.pkl', 'wb') as f:
        pickle.dump(preprocessors['label_encoders'], f)
    with open(f'{Config.MODEL_DIR}feature_names.pkl', 'wb') as f:
        pickle.dump(preprocessors['feature_names'], f)
    logger.info("✅ Saved all model artifacts")
    
    metadata = {
        'model_name': model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_metrics': results_df[results_df['Model'] == model_name].to_dict('records')[0]
    }
    with open(f'{Config.MODEL_DIR}model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info("✅ Saved metadata")

def main():
    print("\n" + "="*80)
    print("🛡️  CHURNGUARD AI - CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*80)
    print("Intelligent ML-powered churn prediction and analysis")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    try:
        Config.create_directories()
        df = load_data(Config.DATA_PATH)
        churn_rate = explore_data(df)
        X_train, X_test, y_train, y_test, preprocessors = preprocess_data(df)
        models = train_models(X_train, y_train)
        results_df, best_model_name = evaluate_models(models, X_test, y_test)
        best_model = models[best_model_name]
        analyze_best_model(best_model, best_model_name, X_test, y_test, preprocessors['feature_names'])
        save_artifacts(best_model, best_model_name, preprocessors, results_df)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Churn Rate: {churn_rate:.2f}%")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()