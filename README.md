# 🛡️ ChurnGuard AI - Customer Churn Prediction System

ML-powered customer churn prediction system with **73.81% accuracy** and **0.84 AUC-ROC**.

## 🎯 Features

- ✅ 4 ML Models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- ✅ 73.81% Accuracy with Logistic Regression
- ✅ 0.84 AUC-ROC Score
- ✅ Complete EDA with 6 visualizations
- ✅ Production-ready prediction system

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Make predictions
python predict.py
```

## 📊 Model Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| **Logistic Regression** ⭐ | **73.81%** | **0.6168** | **0.8398** |
| Random Forest | 77.50% | 0.6062 | - |
| Decision Tree | 78.07% | 0.5830 | - |
| Gradient Boosting | 79.56% | 0.5714 | - |

## 📁 Project Structure
```
churnGuard-AI/
├── data/                  # Dataset (7,043 customers)
├── models/                # Trained models
├── outputs/               # 6 visualizations
├── train_model.py         # Training script
├── predict.py             # Prediction script
└── requirements.txt       # Dependencies
```

## 🎓 Key Insights

- **Contract Type:** Month-to-month customers have 42% churn rate vs 3% for two-year contracts
- **Tenure:** 50% of churn occurs in first 6 months
- **Services:** Customers with 3+ services have 15% churn vs 35% with single service
- **Tech Support:** Customers with tech support have 18% churn vs 35% without

## 💰 Business Impact

**Potential Annual Savings:** $267K - $991K

- 7,043 customers
- 26.54% churn rate
- $80 average monthly revenue
- With 30-50% retention improvement

## 📖 Documentation

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete project details

## 📦 Dependencies
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

## 🔮 Prediction Example
```python
from predict import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict(customer_data)

# Output:
# 📊 Prediction: Churn
# 📈 Churn Probability: 84.21%
# ⚠️  Risk Level: 🔴 HIGH RISK
```

## 🛠️ Technologies

- Python 3.9+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 📝 License

MIT License

## 👤 Author

**Chikle Aniketh**
- GitHub: [@AnikethChikle](https://github.com/AnikethChikle)
- Email: chikleaniketh@gmail.com

---
✅ Now You Have All Files!
Check you have:

✅ README.md (just created)
✅ PROJECT_SUMMARY.md (created earlier)
✅ train_model.py
✅ predict.py
✅ requirements.txt
✅ Models in models/ folder
✅ Visualizations in outputs/visualizations/


🚀 Now Push to GitHub
bash# Add the new README.md
git add README.md

# Commit it
git commit -m "Add README.md documentation"

# Push to GitHub
git push -u origin main
