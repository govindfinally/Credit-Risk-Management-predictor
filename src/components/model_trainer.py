import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_file_path=os.path.join('artifact','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Data got splitted into tarin and test data respecively")
            Xtrain,Ytrain,Xtest,Ytest=(train_array[:, :-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "KNeighbors":KNeighborsRegressor(),
                "LinearRegression":LinearRegression(),
                "AdaBoost":AdaBoostRegressor()
            }
            params={
                "RandomForest":{
                    'n_estimators':[100],
                    'max_features':['auto'],
                    'max_depth':[10,20,30,40,50],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,4]
                },
                "DecisionTree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_depth':[10,20,30,40,50],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,4]
                },
                "GradientBoosting":{
                    'n_estimators':[100],
                    'learning_rate':[0.01,0.1,0.2],
                    'max_depth':[3,5,7],
                    'min_samples_split':[2,5],
                    'min_samples_leaf':[1,2]
                },
                "XGBoost":{
                    'n_estimators':[100],
                    'learning_rate':[0.01,0.1],
                    'max_depth':[3,5],
                    'subsample':[0.8],
                    'colsample_bytree':[0.8]
                },
                "CatBoost":{
                    'iterations': [100],
                    'depth': [6],
                    'learning_rate': [0.1]
                },
                "KNeighbors":{
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "LinearRegression":{},
                "AdaBoost":{
                    "n_estimators":[50],
                    "learning_rate":[1.0]
                }
            }
            model_report:dict=evaluate_models(Xtrain,Ytrain,Xtest,Ytest,models,param=params)    
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
            best_model=models[best_model_name]
            best_model.fit(Xtrain,Ytrain)
            if best_model_score<0.75:
                raise CustomException("No best model, try with another model")
            logging.info("Best model found")
            save_object(
                file_path=self.model_trainer_config.trained_file_path,
                obj=best_model
            )
            prediction=best_model.predict(Xtest)
            r2_square=r2_score(Ytest,prediction)
            return r2_square
        except Exception as e:
            logging.info("Error occured in model trainer")  
            raise CustomException(e,sys)

