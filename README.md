# Customer Retention Analysis - E-commerce Project

A complete end-to-end data science project analyzing customer churn and segmentation for an online retail business.

## 📊 Project Overview

This project analyzes 541,909 transactions from a UK-based online retailer to:
- Identify customer segments using RFM analysis and K-means clustering
- Predict customer churn with 100% accuracy using Random Forest
- Provide actionable business recommendations

## 🎯 Key Findings

- **33.4% churn rate** across customer base
- **4 distinct customer segments** identified
- **217 VIP customers** (5% of base) drive majority of revenue
- **Recency** is the strongest predictor of churn (72% feature importance)

## 📁 Project Structure
```
customer-retention-analysis/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # RFM features, segments
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
├── outputs/
│   ├── figures/                # Visualizations
│   └── models/                 # Trained models
├── README.md
└── requirements.txt
```

## 🚀 How to Run

1. Clone the repository
```bash
git clone <your-repo-url>
cd customer-retention-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run notebooks in order
- `01_data_exploration.ipynb` - Initial data analysis
- `02_feature_engineering.ipynb` - Segmentation and churn prediction

## 📈 Customer Segments

| Segment | Count | Avg Spend | Churn Rate | Action |
|---------|-------|-----------|------------|--------|
| VIP Champions | 13 | £127k | 0% | White-glove service |
| High-Value Loyalists | 204 | £12.7k | 2% | Exclusive perks |
| Loyal Customers | 3,054 | £1.4k | 12% | Retention focus |
| At Risk/Lost | 1,067 | £481 | 100% | Win-back campaigns |

## 🛠️ Technologies Used

- Python 3.10
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization
- K-means clustering, Random Forest

## 💡 Business Impact

- Identified £250k+ in savable revenue from at-risk customers
- Created actionable customer segments for targeted marketing
- Built predictive model to flag at-risk customers proactively

## 📧 Contact

[Your Name] - [Your Email/LinkedIn]

---

**Dataset Source:** UCI Machine Learning Repository - Online Retail Dataset