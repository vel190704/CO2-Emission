from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
with open('co2.pkl', 'rb') as handle:
    model = pickle.load(handle)
data = pd.read_csv('C:/Users/Raghul727/Desktop/CO2-Emission/Indicators.csv')
data = data.drop(['IndicatorCode', 'CountryCode'] ,axis=1) 
data = pd.DataFrame(data)
data["CountryName"] = le1.fit(data["CountryName"])
data["IndicatorName"] = le2.fit(data["IndicatorName"])
app = Flask(__name__)
@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/prediction', methods = ["POST","GET"])
def prediction():
    input_feature = [x for x in request.form.values()]
    features_values = [np.array(input_feature)]
    feature_name = ["CountryName","IndicatorName","Year"]
    x_test = pd.DataFrame(features_values,columns=feature_name)
    x_test["CountryName"] = le1.transform(x_test["CountryName"])
    x_test["IndicatorName"] = le2.transform(x_test["IndicatorName"])
    prediction = model.predict(x_test)
    print("Prediction is ", prediction)
    return render_template("index1.html",prediction_text = str(prediction[0]))

@app.route('/Home', methods=['POST', 'GET'])
def my_home():  
    return render_template('index.html')
    
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)



