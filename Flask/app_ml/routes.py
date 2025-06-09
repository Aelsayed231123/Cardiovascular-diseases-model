from flask import Flask, render_template , request
from app_ml import app
from app_ml.models import modeltf ,loadedpp , knn , lreg
import pandas as pd



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    ChestPainType = request.form['ChestPainType']
    RestingBP = float(request.form['RestingBP'])
    Cholesterol = float(request.form['Cholesterol'])
    FastingBS = request.form['FastingBS']
    RestingECG = request.form['RestingECG']
    MaxHR = float(request.form['MaxHR'])
    ExerciseAngina = request.form['ExerciseAngina']
    Oldpeak = float(request.form['Oldpeak'])
    ST_Slope = request.form['ST_Slope']
    modeltype = request.form['Model']
    data = {
        'Age' : age,
        'Sex' :gender,
        'ChestPainType' : ChestPainType,
        'RestingBP' : RestingBP,
        'Cholesterol' : Cholesterol,
        'FastingBS' : FastingBS,
        'RestingECG' : RestingECG,
        'MaxHR' : MaxHR,
        'ExerciseAngina' : ExerciseAngina,
        'Oldpeak' : Oldpeak,
        'ST_Slope' : ST_Slope
    }
    df = pd.DataFrame(data,index=[0])
    print(df)
    df_scaled = loadedpp.transform(df)
    knnpred = knn.predict(df_scaled)
    lregpred = lreg.predict(df_scaled)



    DLprediction = modeltf.predict(df_scaled)
    print(DLprediction)
    print(knnpred)
    print(lregpred)

    if modeltype == "KNN":
        finalpred = 'Possible Heart Disease' if knnpred== 1 else 'No Heart Disease'
    elif modeltype == "Lreg":
        finalpred = 'Possible Heart Disease' if lregpred == 1 else 'No Heart Disease'
    else:
        finalpred = 'Possible Heart Disease' if DLprediction >=0.5 else 'No Heart Disease'


    print(finalpred)


    return render_template('result.html',predicted_class = finalpred)


if __name__ == '__main__':
    app.run(debug=True)