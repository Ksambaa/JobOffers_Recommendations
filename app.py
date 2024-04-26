import os
import sys
import pickle
import streamlit as st
import numpy as np
from joboffers_recommender.logger.log import logging
from joboffers_recommender.config.configuration import AppConfiguration
from joboffers_recommender.pipeline.training_pipeline import TrainingPipeline
from joboffers_recommender.exception.exception_handler import AppException


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self,suggestion):
        try:
            joboffer_name = []
            ids_index = []
            poster_url = []
            joboffer_pivot =  pickle.load(open(self.recommendation_config.joboffer_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for joboffer_id in suggestion:
                joboffer_name.append(joboffer_pivot.index[joboffer_id])

            for name in joboffer_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            raise AppException(e, sys) from e
        


    def recommend_joboffer(self,joboffer_name):
        try:
            joboffers_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            joboffer_pivot =  pickle.load(open(self.recommendation_config.joboffer_pivot_serialized_objects,'rb'))
            joboffer_id = np.where(joboffer_pivot.index == joboffer_name)[0][0]
            distance, suggestion = model.kneighbors(joboffer_pivot.iloc[joboffer_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    joboffers = joboffer_pivot.index[suggestion[i]]
                    for j in joboffers:
                        joboffers_list.append(j)
            return joboffers_list , poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self,selected_joboffers):
        try:
            recommended_joboffers,poster_url = self.recommend_joboffer(selected_joboffers)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_joboffers[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_joboffers[2])
                st.image(poster_url[2])

            with col3:
                st.text(recommended_joboffers[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_joboffers[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_joboffers[5])
                st.image(poster_url[5])
        except Exception as e:
            raise AppException(e, sys) from e



if __name__ == "__main__":
    st.header('ML Based joboffers Recommender System')
    st.text("This is a collaborative filtering based recommendation system!")

    obj = Recommendation()

    #Training
    if st.button('Train Recommender System'):
        obj.train_engine()

    joboffer_names = pickle.load(open(os.path.join('templates','book_names.pkl') ,'rb'))
    selected_joboffers = st.selectbox(
        "Type or select a joboffer from the dropdown",
        joboffer_names)
    
    #recommendation
    if st.button('Show Recommendation'):
        obj.recommendations_engine(selected_joboffers)
