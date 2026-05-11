import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

df=pd.read_csv("Cleaning&Engineering.csv")
print("data loaded")
# encoding binary cols
le=LabelEncoder()
df["RenewableEnergy"]=le.fit_transform(df["RenewableEnergy"])
# encoding target
df["ConsumptionCategory"]=le.fit_transform(df["ConsumptionCategory"])
# numerical columns
num_cols=["NumOccupants","NumRooms","NumAppliances","BuildingAge(Years)","HoursPowerDaily",
          "SolarCapacity(kW)","PeakHourUsage","NumACUnits","NumFreezers","MonthlyIncome_log",
          "MonthlyBill_log","GeneratorFuelCost_log","MonthlyConsumption_log","consumption_rate",
          "House_comfort","usage_rate","appliance_consumption"]
# categorical columns
cat_cols=["State","Sector","BuildingType","ElectricitySource","TariffBand","HasMeter",
          "BackupPower"]
# binary columns
bin_cols=["RenewableEnergy"]
x=df[num_cols + cat_cols + bin_cols]
y=df["ConsumptionCategory"]
# Transformers/pipeline
preprocessor=ColumnTransformer(transformers=[
    ("Scaler", StandardScaler(), num_cols),
    ("ohe", OneHotEncoder(drop='first',
                          sparse_output=False,
                          handle_unknown="ignore"), cat_cols)
], remainder="passthrough")
# splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.25,
                                               random_state=7,
                                               stratify=y)
# preprocessing
x_train_P=preprocessor.fit_transform(x_train)
x_test_P=preprocessor.transform(x_test)
# saving preprocessor
joblib.dump(preprocessor,"preprocessor.joblib")
print("PreprocessorsSaved")
# model
model=XGBClassifier(learning_rate=0.2,
                    max_depth=8,
                    n_estimators=150,
                    subsample=0.7,
                    colsubsample_bytree=0.8,
                    eval_metric='mlogloss',
                    use_label_encoder=False,
                    random_state=7,
                    n_jobs=-1
                    )
model.fit(x_train_P, y_train)
y_pred=model.predict(x_test_P)
accuracy=accuracy_score(y_test, y_pred)
report=classification_report(y_test, y_pred)
joblib.dump(model, 'XGB_model.joblib')
print(f"Accuracy: {accuracy*100:.2f}%")
print(report)
LGBM_model=LGBMClassifier(
    n_estimators=300,
    max_depth=11,
    num_leaves=64,
    learning_rate=0.08,
    subsample=0.8,
    random_state=7,
    n_jobs=-1
)
LGBM_model.fit(x_train_P, y_train)
y_pred1=LGBM_model.predict(x_test_P)
accuracy1=accuracy_score(y_test, y_pred1)
report1=classification_report(y_test, y_pred1)
joblib.dump(LGBM_model, 'LGBM_model.joblib')
print(f"Accuracy: {accuracy1*100:.2f}%")
print(report1)
