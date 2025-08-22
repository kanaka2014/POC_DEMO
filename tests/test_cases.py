from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
import pandas as pd

# Assert data set shape

def test_load_df():
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    assert train_data is not None
    assert test_data is not None
    df_train= pd.read_csv(train_data)
    df_test= pd.read_csv(test_data)
    assert df_train.shape == (800, 8)
    assert df_test.shape == (200, 8)  

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    assert train_arr.shape==(800, 20)
    assert test_arr.shape==(200, 20)


    
