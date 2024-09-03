import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('Artifact', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1] 

            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            models = {
                    'LogisticRegression': LogisticRegression(),
                    'GaussianNB':GaussianNB(),
                    'SVC':SVC(),
                    'KNeighborsClassifier':KNeighborsClassifier(),
                    'DecisionTreeClassifier':DecisionTreeClassifier(),
                    'ExtraTreeClassifier':ExtraTreeClassifier(),
                    'RandomForestClassifier':RandomForestClassifier(),
                    'BaggingClassifier':BaggingClassifier(),
                    'GradientBoostingClassifier':GradientBoostingClassifier(),
                    'AdaBoostClassifier':AdaBoostClassifier()
            }


            model_report:dict = evaluate_models(x_train=x_train, y_train = y_train, x_test=x_test, y_test=y_test, models= models)


            ##best_model_score

            best_model_score = max(sorted(model_report.values()))

            ## best model name

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )

            y_test_predict = best_model.predict(x_test)

            acc_score = accuracy_score(y_test, y_test_predict)

            return acc_score, best_model_name

        except Exception as e:
            raise CustomException(e,sys)