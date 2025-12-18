import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
import dill
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_pt:
            dill.dump(obj,file_pt)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_modelm(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = r2_score(y_test, y_pred)

            report[name] = {
                "score": score,
                "model": model  
            }
        return report

      
    except Exception as e:
        raise CustomException(e,sys)
    
    