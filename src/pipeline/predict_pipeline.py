import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("Artifact","model.pkl")
            preprocessor_path=os.path.join('Artifact','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        N: float,
        P: float,
        K:float,
        temperature:float,
        humidity: float,
        ph: float,
        rainfall: float):

        self.N = N

        self.P = P

        self.K =  K

        self.temperature = temperature

        self.humidity = humidity

        self.ph = ph

        self.rainfall = rainfall

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "N": [self.N],
                "P": [self.P],
                "K": [self.K],
                "temperature": [self.temperature],
                "humidity": [self.humidity],
                "ph": [self.ph],
                "rainfall": [self.rainfall]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)