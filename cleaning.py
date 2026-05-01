import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("data/nigerian_energy_consumption_messy.csv")
df=df.drop_duplicates()
df["State"]=df["State"].astype(str).str.strip()
df["Sector"]=df["Sector"].astype(str).str.strip()
df["BuildingType"]=df["BuildingType"].astype(str).str.capitalize().str.strip()

df["ElectricitySource"]=df["ElectricitySource"].astype(str).str.capitalize().str.strip()
electricitysource_corrector={"Nepa":"NEPA/PHCN","Nepa/phcn":"NEPA/PHCN","Grid":"NEPA/PHCN","Off-grid":"Solar","Mixed":"Both"}
df["ElectricitySource"]=df["ElectricitySource"].replace(electricitysource_corrector)
df["TariffBand"]=df["TariffBand"].astype(str).str.upper().str.strip()
def Tariff(c):
    if "AND" in c:
        return c.replace("AND "," and ")
    else:
        return "B and "+c
df["TariffBand"]=df["TariffBand"].apply(lambda x: Tariff(x))
#HasMeter&RenewableEnergy
for c in ["HasMeter","RenewableEnergy"]:
    df[c]=df[c].astype(str).str.capitalize().str.strip()
    corrector={"N":"No","Y":"Yes","0":"No","1":"Yes"}
df[c]=df[c].replace(corrector)
#BackupPower
df["BackupPower"]=df["BackupPower"].astype(str).str.capitalize().str.strip()
Backup_corrector={"Ups":"UPS","Gen":"Generator","None":"No Backup","No backup":"No Backup","Nan":np.nan}
df["BackupPower"]=df["BackupPower"].replace(Backup_corrector)
df["BackupPower"]=df["BackupPower"].fillna(df.groupby(["Sector","BuildingType","ElectricitySource"])["BackupPower"].transform(lambda x: x.mode()[0]))
# monthly income,monthly bill, generator fuel cost
for c in ["MonthlyIncome(NGN)","MonthlyBill(NGN)","GeneratorFuelCost(NGN)"]:
    df[c]=df[c].astype(str).str.replace("NGN","").str.replace(",","").str.replace("\u20A6","").str.replace("-","")
    df[c]=pd.to_numeric(df[c],errors="coerce")
    max1=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min1=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].apply(lambda x: np.nan if x>max1 or x<min1 else x)
df["MonthlyIncome(NGN)"]=(df["MonthlyIncome(NGN)"].fillna(df.groupby(["Sector","BuildingType","BackupPower"])["MonthlyIncome(NGN)"].transform("mean"))).round(1)
df["MonthlyBill(NGN)"]=df["MonthlyBill(NGN)"].fillna(df.groupby(["State","Sector","BuildingType"])["MonthlyBill(NGN)"].transform("mean")).round(1)
df["GeneratorFuelCost(NGN)"]=df["GeneratorFuelCost(NGN)"].fillna(df.groupby(["State","ElectricitySource","ElectricitySource"])["GeneratorFuelCost(NGN)"].transform("mean")).round(1)
# building age
df["BuildingAge(Years)"]=df["BuildingAge(Years)"].astype(str).str.replace("yrs","").str.replace("years","").str.replace("-","")
df["BuildingAge(Years)"]=pd.to_numeric(df["BuildingAge(Years)"],errors="coerce")
df["BuildingAge(Years)"]=(df["BuildingAge(Years)"].fillna(df.groupby(["State","Sector"])["BuildingAge(Years)"].transform("median")))
df["BuildingAge(Years)"]=df["BuildingAge(Years)"].astype(int)
# hours power daily
df["HoursPowerDaily"]=df["HoursPowerDaily"].astype(str).str.replace("-","").str.replace("hours","").str.replace("hrs","").str.strip()
df["HoursPowerDaily"]=pd.to_numeric(df["HoursPowerDaily"], errors="coerce")
df["HoursPowerDaily"]=df["HoursPowerDaily"].apply(lambda x: np.nan if x>24 else x)
df["HoursPowerDaily"]=df["HoursPowerDaily"].fillna(df.groupby(["Sector","ElectricitySource","BackupPower"])["HoursPowerDaily"].transform("mean")).round(1)
# solar capacity
df["SolarCapacity(kW)"]=df["SolarCapacity(kW)"].astype(str).str.replace("kW","").str.replace("-","").str.strip()
df["SolarCapacity(kW)"]=pd.to_numeric(df["SolarCapacity(kW)"],errors="coerce")
df["SolarCapacity(kW)"]=df.apply(lambda x: 0 if x["RenewableEnergy"]=="No" else x["SolarCapacity(kW)"],axis=1)
df["SolarCapacity(kW)"]=df["SolarCapacity(kW)"].fillna(df.groupby("Sector")["SolarCapacity(kW)"].transform("mean")).round(1)
# monthly concumption
df["MonthlyConsumption(kWh)"]=df["MonthlyConsumption(kWh)"].astype(str).str.replace("kh","").str.replace("w","").str.replace("W","").str.replace("-","").str.strip()
df["MonthlyConsumption(kWh)"]=pd.to_numeric(df["MonthlyConsumption(kWh)"], errors="coerce")
sns.boxplot(df["MonthlyConsumption(kWh)"])
df["MonthlyConsumption(kWh)"]=df["MonthlyConsumption(kWh)"].fillna(df.groupby(["Sector","BackupPower"])["MonthlyConsumption(kWh)"].transform("mean")).round(1)
print(df["MonthlyConsumption(kWh)"].isnull().sum())
