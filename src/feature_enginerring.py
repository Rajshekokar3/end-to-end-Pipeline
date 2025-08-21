import os 
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


log_dir="logs"
os.makedirs(log_dir,exist_ok=True) # it present or not

#logging configuration 
logger =logging.getLogger('Feature Engineering')
logger.setLevel("DEBUG")

console_handler =logging.StreamHandler() # it will print in terminal 
console_handler.setLevel("DEBUG")

#file handler 
log_file_path=os.path.join(log_dir,"Feature Enginerring.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


#Formating 
formattter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formattter)# applying the formatter
file_handler.setFormatter(formattter) # applying the formatter 


logger.addHandler(console_handler)
logger.addHandler(file_handler)


#yamal funcition
def load_params(params_path:str)-> dict:
    """ LOad parameters from YAML file. """
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexcepted error: %s",e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """
    load data from csv file
    """
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("Data loadedand nans filled from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse th csv file %s",e)
        raise
    except Exception as e:
        logging.error("Unexcepted error occured while looking the data:%s",e)
        raise 

def apply_tfid(train_data:pd.DataFrame,test_data:pd.DataFrame,max_feature:int)->tuple:

    """
    Apply tfid to the data

    """
    try:
        vectorizer=TfidfVectorizer(max_features=max_feature)

        x_train=train_data['text'].values
        y_train=train_data['target'].values
        
        x_test=test_data['text'].values
        y_test=test_data['target'].values

        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)
        
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        logger.debug("Bag of words applied and data transformed")
        return train_df,test_df
    except Exception as e:
        logger.error("Error during Bag of Words transformed %s",e)


def save_data(df:pd.DataFrame,file_path:str) ->None:
    """
      Save the dataframe to a CSV FILE.
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False,mode="w")
        logger.debug('Data saved to %s',file_path)

    except Exception as e:
        logger.error("Unexcepted error occured while saving the data %s",e)
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        max_features=params['feature_enginerring']['max_features']
        #max_feature=50

        train_data=load_data('./data/interim/train_processed_data.csv')
        test_data=load_data('./data/interim/test_processed_data.csv')

        train_df,test_df=apply_tfid(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed","train_tfid.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tfid.csv"))

    except Exception as e:
        logger.error("Failed to complete the feature enginerring process: %s",e)
        print(f"Error",e)


if __name__=="__main__":
    main()