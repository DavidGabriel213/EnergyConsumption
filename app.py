from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import joblib
import os

model=joblib.load('LGBM_model.joblib')
preprocessor=joblib.load('preprocessor.joblib')
print('model and preprocessor loaded')
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def myfunc():
    category=None
    cat_class=None
    if request.method=='POST':
        # numerical columns
        NumOccupants=float(request.form['occupants'])
        NumRooms=float(request.form['rooms'])
        NumAppliances=float(request.form['appliances'])
        BuildingAge=float(request.form['age'])
        HoursPowerDaily=float(request.form['hour_usage'])
        SolarCapacity=float(request.form['solar_capacity'])
        PeakHourUsage=float(request.form['peak_usage'])
        NumACUnits=float(request.form['num_AC'])
        NumFreezers=float(request.form['num_freezers'])
        MonthlyIncome=float(request.form['income'])
        MonthlyBill=float(request.form['bill'])
        GeneratorFuelCost=float(request.form['fuel_cost'])
        MonthlyConsumption=float(request.form['monthly_consumption'])
        # categorical columns
        State=request.form['state']
        Sector=request.form['sector']
        BuildingType=request.form['building_type']
        ElectricitySource=request.form['source']
        TariffBand=request.form['tarrifBand']
        HasMeter=request.form['has_meter']
        BackupPower=request.form['backup']
        # binary columns
        RenewableEnergy=float(request.form['renewable_energy'])
        # feature engineering
        MonthlyIncome_log=np.log1p(MonthlyIncome)
        MonthlyBill_log=np.log1p(MonthlyBill)
        GeneratorFuelCost_log=np.log1p(GeneratorFuelCost)
        MonthlyConsumption_log=np.log1p(MonthlyConsumption)
        consumption_rate=(MonthlyConsumption_log/NumOccupants)
        House_comfort=(MonthlyBill_log/(NumRooms+NumAppliances))
        usage_rate=(MonthlyIncome_log/(MonthlyBill_log+GeneratorFuelCost_log))
        appliance_consumption=(HoursPowerDaily/(NumRooms+NumAppliances))
        # features
        feature=pd.DataFrame({"NumOccupants":[NumOccupants],"NumRooms":[NumRooms],"NumAppliances":[NumAppliances],
                              "BuildingAge(Years)":[BuildingAge],"HoursPowerDaily":[HoursPowerDaily],"SolarCapacity(kW)":[SolarCapacity],
                              "PeakHourUsage":[PeakHourUsage],"NumACUnits":[NumACUnits],"NumFreezers":[NumFreezers],
                              "MonthlyIncome_log":[MonthlyIncome_log],"MonthlyBill_log":[MonthlyBill_log],
                              "GeneratorFuelCost_log":[GeneratorFuelCost_log],"MonthlyConsumption_log":[MonthlyConsumption_log],
                              "consumption_rate":[consumption_rate], "House_comfort":[House_comfort],
                              "usage_rate":[usage_rate],"appliance_consumption":[appliance_consumption],
                              "State":[State],"Sector":[Sector],"BuildingType":[BuildingType],"ElectricitySource":[ElectricitySource],
                              "TariffBand":[TariffBand],"HasMeter":[HasMeter],"BackupPower":[BackupPower],"RenewableEnergy":[RenewableEnergy]
                              })
        # preprocessing
        FEATURES=preprocessor.transform(feature)
        prediction=model.predict(FEATURES)[0]
        if prediction==0:
            category="High"
        elif prediction==1:
            category="Low"
        elif prediction==2:
            category="Moderate"
        else:
            category="Very high"
        class_map={
            'Low': 'result-low',
            'Moderate': 'result-moderate',
            'High': 'result-high',
            'Very high': 'result-veryhigh'   
        }
        cat_class=class_map.get(category, '')
    return render_template("front.html", category=category, cat_class=cat_class)
if __name__==('__main__'):
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)   
