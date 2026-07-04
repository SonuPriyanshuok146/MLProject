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

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier":AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error','absolute_error','poisson'],
                    # 'splitter': ['best','random'],
                    # 'max_features': ['sqrt','log2'],
                    # 'max_depth': [None, 5, 10, 20],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5],
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256],
                    # 'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features': ['sqrt','log2'],
                    # 'max_depth': [None, 5, 10, 20],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5],
                    # 'bootstrap': [True, False],
                },
                "Gradient Boosting": {
                    'learning_rate': [.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'criterion': ['squared_error','friedman_mse'],
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "Linear Regression": {
                    # 'fit_intercept': [True, False],
                    # 'copy_X': [True, False],
                    # 'positive': [True, False],
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [5, 7,9, 11],
                    # 'weights': ['uniform','distance'],
                    # 'algorithm': ['auto','ball_tree','kd_tree','brute'],
                    # 'leaf_size': [20, 30, 40],
                    # 'p': [1, 2],
                },
                "XGBClassifier": {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_depth': [3, 5, 7],
                    # 'subsample': [0.8, 1.0],
                    # 'colsample_bytree': [0.8, 1.0],
                    # 'gamma': [0, 1, 5],
                    # 'reg_alpha': [0, 0.1, 1],
                    # 'reg_lambda': [1, 1.5, 2],
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30,50,100],
                    # 'l2_leaf_reg': [1, 3, 5],
                    # 'border_count': [32, 64, 128],
                    # 'grow_policy': ['SymmetricTree','Depthwise','Lossguide'],
                },
                "AdaBoost Classifier": {
                    'learning_rate': [.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'loss': ['linear','square','exponential'],
                }
}

            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testig dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(x_test)
            
            r2_square = r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)