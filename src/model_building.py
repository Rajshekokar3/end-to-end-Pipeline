import os
import numpy as np 
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

#Ensure the "logs directory exists"
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('model_building')
logger.setLevel("DEBUG")

# console handeler 
console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

#file handler 
log_file_path=os.path.join(log_dir,'model_building.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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
    Load data from csv file.
    :param file_path:Path to the CSV file 
    :return :Loaded Dataframe
    
    """
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("FILe NOT FOUND %s",e)
        raise
    except Exception as e:
        logger.error("Unexcepted eroor ocurred while loading the data: %s",e)
        raise 


def train_model(X_train:np.ndarray,y_train:np.ndarray,params: dict) -> RandomForestClassifier: #this coding style is type hunting just for the coding standard
    #Defining all the data types which we are passsing and -> what we are returning is Random Forest Classifier 
    """
    Train the RandomForest Model.

    :param X_train:Training features
    :param Y_train:Traning labels
    :Parma Param:Dictinoary of hyperparmeters
    :Return :Trained RandomForestClassifier

    """

    try:
        if X_train.shape[0] !=y_train.shape[0]:
            raise ValueError("The number of sample in X_train and y_train must be the same")
        
        logger.debug("Initalizig the RandomForest model with parameters: %s",params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug("Model traning started with %s samples", X_train.shape[0])
        clf.fit(X_train,y_train)

        logger.debug('Model traning completed')

        return clf
    except ValueError as e:
        logger.error('ValueError during model traning: %s',e)
        raise
    except Exception as e:
        logger.error("Error during model traning: %s",e)
        raise


def save_model(model,file_path: str)-> None:
    """
    Save the tranined model to a file.
    :parm model: Trained model object 
    :parm file_path: Path to save the model fits
    
    """
    try:
        #Ensure the direcotory exits

        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved to %s",file_path)

    except FileNotFoundError as e:
        logger.error("File path not Found: %s",e)
        raise
    except Exception as e:
        logger.error("Error ocurred while saving model : %s",e)
        raise

def main():
    try:
        params=load_params('params.yaml')['model_building'] #
        #params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed/train_tfid.csv')
        X_train=train_data.iloc[:, :-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(X_train,y_train,params)

        model_save_path= 'models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s",e)
        print(f"Error: {e}")

if __name__=="__main__":
    main()