# ⚡ Nigerian Energy Consumption Predictor

A complete multi-class ML project predicting household energy consumption category — **Low, Moderate, High or Very High** — trained on 20,080 Nigerian household records. All 6 models trained including Neural Network. XGBoost deployed.

## 🌐 Live Demo
**[Try the app →](https://energyconsumption-xydx.onrender.com)**

---

## 📌 The Honest Deployment Story

The Neural Network achieved **77.69%** — the best result across all 6 models. It was NOT deployed.

**Why?**

TensorFlow requires ~500MB RAM to load. Free hosting platforms provide ~512MB total. The app crashed on startup every time. LightGBM also encountered dependency conflicts on the deployment server.

**XGBoost was deployed** — 8MB, loads in milliseconds, never crashes.

> *The best model for production is not always the most accurate model in isolation. That is real engineering.*

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 20,080 |
| Columns | 23 |
| Target | ConsumptionCategory: 4 classes |
| Classes | Low / Moderate / High / Very High |

---

## 🤖 All 6 Models Trained
| Model | Accuracy | Status |
|---|---|---|
| Logistic Regression | 73.97% | Not deployed |
| Decision Tree | 54.96% | Not deployed |
| Random Forest | 65.50% | Not deployed |
| **XGBoost** | **73.57%** | **DEPLOYED ✅** |
| LightGBM | 73.67% | Failed on server |
| **Neural Network** | **77.69%** | **Best accuracy — RAM limit ❌** |

### Key Finding
This is the first project where Neural Network outperformed ALL traditional models. Energy consumption has complex non-linear interactions between AC units, income, sector and hours of power — exactly where NNs outshine trees.

---

## 🧹 Data Cleaning
| Column | Problem | Solution |
|---|---|---|
| MonthlyBill | "NGN18,500", "₦22,000", outliers x100 | Strip NGN/₦, commas, IQR clip |
| MonthlyIncome | Same currency chaos | Same pipeline |
| GeneratorFuelCost | Same currency chaos | Same pipeline |
| MonthlyConsumption | "350 kWh", "350kwh", outliers x10 | Strip kWh, IQR clip |
| HoursPowerDaily | "10 hrs", "10 hours", negatives | Strip, np.abs(), clip 0-24 |
| SolarCapacity | Mixed formats + logic fill | If RenewableEnergy=No → 0 |
| BuildingAge | "12 years", "12yrs", negatives | Strip, np.abs() |
| ConsumptionCategory | 28 different formats | str.capitalize() + dict map |
| Duplicates | 80 hidden rows | drop_duplicates() |

---

## ⚙️ Feature Engineering
| Feature | Formula | Meaning |
|---|---|---|
| consumption_rate | MonthlyConsumption_log / NumOccupants | Per-person energy consumption |
| House_comfort | MonthlyBill_log / (NumRooms + NumAppliances) | Bill relative to household size |
| usage_rate | MonthlyIncome_log / (Bill_log + FuelCost_log) | Energy affordability ratio |
| appliance_consumption | HoursPowerDaily / (NumRooms + NumAppliances) | Power hours per appliance |
| Hourly_rate | MonthlyConsumption_log / HoursPowerDaily | kWh per hour of power available |

---

## 🧠 Neural Network Architecture
```python
model = keras.Sequential([
    Dense(256, activation='relu', input_shape=(n,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
```
Trained in Google Colab. Achieved 77.69%. Not deployed due to RAM constraints on free hosting.

---

## 🏗️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Deep Learning:** TensorFlow/Keras (trained, not deployed)
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3 (Dark Electric Theme)
- **Deployment:** Railway.app
- **Version Control:** GitHub

---

## 📁 Project Structure
```
EnergyConsumptionPredictor/
├── data/
│   └── nigerian_energy_consumption_messy.csv
├── models/
│   ├── xgb_model.joblib
│   ├── preprocessor.joblib
│   └── label_encoder.joblib
├── templates/
│   └── front.html
├── static/
│   └── energy_style.css
├── app.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/EnergyConsumptionPredictor
cd EnergyConsumptionPredictor
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Learnings
1. **NN beats trees on non-linear data** — first project confirming this across 6 models
2. **Production != best accuracy** — RAM constraints make TensorFlow unsuitable for free hosting
3. **Logic-based null fill** — SolarCapacity → 0 when no renewable energy, not mean fill
4. **LightGBM server conflicts** — worked locally, failed in deployment environment
5. **Log transform before feature engineering** — more stable ratio features from normalized values
6. **usage_rate** — energy affordability ratio captures income vs spending relationship

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — This project closes the tabular ML chapter. Next: NLP.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)

