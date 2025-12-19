import sys
import os
import pandas as pd
from src.exception import logging
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
         pass
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts/model.pkl")
            preprocessor_path=os.path.join("artifacts/preprocessor.pkl")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,gender:str,race_enthinicity:str,parental_level_of_education:str,lunch:str,test_prepration_course:str,
                 reading_score:int,writing_score:int
                 ):
        self.gender=gender
        self.race_enthinicity=race_enthinicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_prepration_course=test_prepration_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def get_data_data_df(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race/ethnicity":[self.race_enthinicity],
                "parental level of education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_prepration_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        
             