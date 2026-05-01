import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
#Loading data
df=pd.read_csv("/storage/emulated/0/Download/EnergyConsumption/Cleaning&Engineering.csv")
#Loading preprocessor
preprocessor=joblib.load("/storage/emulated/0/Download/EnergyConsumption/preprocessor.joblib")

le=LabelEncoder()
df["RenewableEnergy"]=le.fit_transform(df["RenewableEnergy"])
# encoding target
df["ConsumptionCategory"]=le.fit_transform(df["ConsumptionCategory"])
# numerical columns
num_cols=["NumOccupants","NumRooms","NumAppliances","BuildingAge(Years)","HoursPowerDaily","SolarCapacity(kW)","PeakHourUsage","NumACUnits","NumFreezers","MonthlyIncome_log","MonthlyBill_log", "GeneratorFuelCost_log","MonthlyConsumption_log","consumption_rate","House_comfort","usage_rate","appliance_consumption"]
# categorical columns
cat_cols=["State","Sector","BuildingType","ElectricitySource","TariffBand","HasMeter","BackupPower"]
# binary columns
bin_cols=["RenewableEnergy"]
x=df[num_cols + cat_cols + bin_cols]
y=df["ConsumptionCategory"]
# splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=7,  stratify=y)
# preprocessing
x_train_P=preprocessor.transform(x_train)
x_test_P=preprocessor.transform(x_test)
#LogisticRegression
model=LogisticRegression(max_iter=1000,random_state=7)
model.fit(x_train_P,y_train)
y_pred=model.predict(x_test_P) 
print(f"LogisticRegression: {accuracy_score(y_test,y_pred)*100:.2f}%")
print(classification_report(y_test,y_pred))
joblib.dump(model,"/storage/emulated/0/Download/EnergyConsumption/LogisticRegreModel.joblib")
#DecisionTreeClassifier
model1=DecisionTreeClassifier(max_depth=12,random_state=7)
model1.fit(x_train_P,y_train)
y_pred1=model1.predict(x_test_P) 
print(f"DecisionTree: {accuracy_score(y_test,y_pred1)*100:.2f}%")
print(classification_report(y_test,y_pred1))
joblib.dump(model1,"/storage/emulated/0/Download/EnergyConsumption/DecisionTreeModel.joblib")
#RandomForestClassifier
model2=RandomForestClassifier(n_estimators=200,random_state=7,max_depth=11)
model2.fit(x_train_P,y_train)
y_pred2=model2.predict(x_test_P) 
print(f"RandomForest: {accuracy_score(y_test,y_pred2)*100:.2f}%")
print(classification_report(y_test,y_pred2))
joblib.dump(model2,"/storage/emulated/0/Download/AgricultureYield/RandomForeModel.joblib")
#feature importance
feature_names = preprocessor.get_feature_names_out()
importances = model2.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_names,'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
feat_imp['Original_Feature'] = feat_imp['Feature'].apply(
    lambda x: x.split('__')[-1].split('_')[0]
)
grouped = feat_imp.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)
print(grouped)
#fine tunning
params = {"C": [0.01, 0.1, 1, 10, 100],"solver": ["liblinear", "saga"],"max_iter": [1000]}
grid=GridSearchCV(LogisticRegression(), params,cv=5, scoring="accuracy")
grid.fit(x_train_P,y_train)
print(f"accuracyscore {grid.best_score_*100:.2f}%")
print(f"bestparameter: {grid.best_params_}")
y_pred3=grid.best_estimator_.predict(x_test_P)
accuracy3=accuracy_score(y_test,y_pred3)
classification3=classification_report(y_test,y_pred3)
print(f"FineTunnedLogistic(accuracy): {accuracy3*100:.2f}%")
print(f"FinetunnedRandomForest(classifcationreport): {classification3}")
joblib.dump(grid.best_estimator_,"/storage/emulated/0/Download/EnergyConsumption/TunnedLogisticModel.joblib")