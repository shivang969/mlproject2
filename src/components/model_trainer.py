import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_modelm


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }
            
            param_grids = {
                "LinearRegression": {
                    "fit_intercept": [True, False]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },

                "DecisionTree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0]
                }
            }


            model_report = evaluate_modelm(
                x_train, y_train, x_test, y_test, models,param_grids
            )

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["score"]
            )

            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]
            best_parms=model_report[best_model_name]['best_parm']
            
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best model: {best_model_name}, R2 Score: {best_model_score} Params: {best_parms}"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
