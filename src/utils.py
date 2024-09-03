from src.exception import CustomException
import sys
import os
import pickle
from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report ={}

        for i in range(len(list(models.keys()))):

            model = list(models.values())[i]

            model.fit(x_train, y_train)

            y_train_predict = model.predict(x_train)

            y_test_predict = model.predict(x_test)

            train_model_score = accuracy_score(y_train, y_train_predict)
            test_model_score = accuracy_score(y_train, y_train_predict)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    


def load_object (file_path):
    try:
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)
    except Exception as e:
        raise CustomException(e, sys)
    