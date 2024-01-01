import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train , X_test, y_test,models:dict):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            logging.info("Model Training Strates")
            model.fit(X_train,y_train)
            logging.info("Training Finished")

            logging.info("Predicting Training data")
            y_train_pred = model.predict(X_train)

            logging.info("Predicting Testing data")
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e :
        logging.info("Error occured in model training")
        raise CustomException(e,sys)


def load_obj(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_object in utils")
        raise CustomException(e,sys)


    # mae = mean_absolute_error(true,predicted)
    # mse = mean_squared_error(true,predicted)
    # rmse = np.sqrt(mean_squared_error(true,predicted))
    # r2_square = r2_score(true,predicted)
    # return mae,rmse,r2_square