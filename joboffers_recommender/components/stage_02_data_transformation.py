import os
import sys
import pickle
import pandas as pd
from joboffers_recommender.logger.log import logging
from joboffers_recommender.config.configuration import AppConfiguration
from joboffers_recommender.exception.exception_handler import AppException



class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    def get_data_transformer(self):
        try:
            df = pd.read_csv(self.data_transformation_config.clean_data_file_path)
            # Lets create a pivot table
            joboffer_pivot = df.pivot_table(columns='user_id', index='title', values= 'rating')
            logging.info(f" Shape of joboffer pivot table: {joboffer_pivot.shape}")
            joboffer_pivot.fillna(0, inplace=True)

            #saving pivot table data
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            pickle.dump(joboffer_pivot,open(os.path.join(self.data_transformation_config.transformed_data_dir,"transformed_data.pkl"),'wb'))
            logging.info(f"Saved pivot table data to {self.data_transformation_config.transformed_data_dir}")

            #keeping joboffers name
            joboffer_names = joboffer_pivot.index

            #saving joboffer_names objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(joboffer_names,open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_names.pkl"),'wb'))
            logging.info(f"Saved joboffer_names serialization object to {self.data_validation_config.serialized_objects_dir}")

            #saving joboffer_pivot objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(joboffer_pivot,open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_pivot.pkl"),'wb'))
            logging.info(f"Saved joboffer_pivot serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



