import os
import sys
from src.logger import logging
from src.exception import CustomException
import dill
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_pt:
            dill.dump(obj,file_pt)
            
    except Exception as e:
        raise CustomException(e,sys)
           