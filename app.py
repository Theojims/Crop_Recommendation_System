from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)
@app.route('/')
@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            N=request.form.get('N'),
            P=request.form.get('P'),
            K=request.form.get('K'),
            temperature=request.form.get('temperature'),
            humidity=request.form.get('humidity'),
            ph=request.form.get('ph'),
            rainfall=request.form.get('rainfall')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        ##print("Before Prediction")

        predict_pipeline=PredictPipeline()
        ##print("Mid Prediction")
        prediction=predict_pipeline.predict(pred_df)
        ##print("after Prediction")


        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to cultivate in this type of soil."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        return render_template('home.html',result = result)
           


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)