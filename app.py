from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.pipeline_predict import CustomData,PredictPipeline



## Route for a home page


data=CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=100,
    writing_score=100

)
pred_df=data.get_data_as_data_frame()
print("preprocess data",pred_df)


predict_pipeline=PredictPipeline()

results=predict_pipeline.predict(pred_df)
print("after Prediction")
print("score is ---",results)
print("hello world")



