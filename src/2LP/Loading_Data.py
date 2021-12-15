import struct
from array import array
import sys
import numpy as np
from sklearn import model_selection
import pandas as pd
import sqlite3 as sql
import os
import struct
from array import array
import sys
import numpy as np

# # Loading Data from idx files to a database

class DataLoader(object):
    """Documentation for DataLoader
 
    DataLoader Class is used to access the files downloaded from MNIST website
    """
    def __init__(self):
        """constructor"""
        self.path = "../../datasets/"
        ## test data path
        self.test_img_path = self.path + 't10k-images-idx3-ubyte'
        ## test data response variables path
        self.test_lbl_path = self.path + 't10k-labels-idx1-ubyte'
        ## train data path
        self.train_img_path = self.path + 'train-images-idx3-ubyte'
        ## train data response variables path
        self.train_lbl_path = self.path + 'train-labels-idx1-ubyte'
        ## test data
        self.test_images = []
        ## test labels
        self.test_labels = []
        ## train data
        self.train_images = []
        ##train images
        self.train_labels = []

    def load_testing(self):
        """load_testing is used to load test images"""
        
        ims, labels = self.load(self.test_img_path,self.test_lbl_path)
        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        """load_testing is used to load test images"""
        
        ims, labels = self.load(self.train_img_path,self.train_lbl_path)
        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        """load checks the magic number of files in paths path_img, path_lbl"""
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('We got an incorrect Magic Number,''got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('We got an incorrect Magic Number,''got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

# ### One Time Creation of Database



#Inheritance
class DatabaseLoader(DataLoader):
    """Documentation for DatabaseLoader
 
    DatabaseLoader Class inherits DataLoader
    DatabaseLoader Class is used to create,access and store data into database
    """
    def __init__(self):
        """constructor"""
        super().__init__()
        ## Datapase path
        self.database = 'database/DigitPrediction.db'
        ## Database connection
        self.conn=sql.connect(self.database)
    
    #Method overriding
    def load_training(self):
        """Method overriding done to return train images and labels"""
        img_train, labels_train = DataLoader.load_training(self)
        img_train = np.array(img_train)
        labels_train = np.array(labels_train)
        return img_train, labels_train
        
    #Method overriding
    def load_testing(self):
        """Method overriding done to return train images and labels"""
        img_test, labels_test = DataLoader.load_testing(self)
        img_test = np.array(img_test)
        labels_test = np.array(labels_test)
        return img_test, labels_test
        
        
    def create_table(self):
        """Cretes tables for train and test set"""
        self.X_train,self.y_train= self.load_training()
        
        self.X_test,self.y_test=self.load_testing()

        #Create table
        #CAN BE DONE ONLY ONCE,NEED TO ADD CODE FOR CONDITIONAL CHECK
        try:
            X_train=pd.DataFrame(self.X_train)
            X_train.to_sql('x_train_db', self.conn)
        except:
            print("Table x_train_db already exists")
        try:
            X_test=pd.DataFrame(self.X_test)
            X_test.to_sql('x_test_db', self.conn)
        except:
            print("Table x_test_db already exists")
            
        try:
            y_train=pd.DataFrame(self.y_train)
            y_train.to_sql('y_train_db', self.conn)
        except:
            print("Table y_train_db already exists")
        try:
            y_test=pd.DataFrame(self.y_test)
            y_test.to_sql('y_test_db', self.conn)
        except:
            print("Table y_test_db already exists")
        
            
    def save_result(self,y_pred):
        """If prediction is not stored, creates a table and stores predictions[y_pred] """
        try:
            y_pred=pd.DataFrame(self.y_pred)
            y_pred.to_sql('y_pred_db', self.conn)
        except:
            #print("Table y_pred_db already exists")  
            pass
        
        
# # Loading from database to dataframe


class DataFrameLoader:
    def __init__(self):
        """Constructor"""
        db = 'database/DigitPrediction.db'
        conn=sql.connect(db)
        
        # Load Dataframes
        try:
            x_train_query="SELECT * from x_train_db"
            ## train data features
            self.x_train=pd.read_sql_query(x_train_query,conn)
            x_test_query="SELECT * from x_test_db"
            ## test data features
            self.x_test=pd.read_sql_query(x_test_query,conn)
            y_train_query="SELECT * from y_train_db"
            ## train data response variables
            self.y_train=pd.read_sql_query(y_train_query,conn)
            y_test_query="SELECT * from y_test_db"
            ## test data response variables
            self.y_test=pd.read_sql_query(y_test_query,conn)
        except:
            print('tables does not exist, need to create tables and stpre data first')

    def load_dataframes(self):
        return self.x_train,self.x_test,self.y_train,self.y_test


if __name__=="__main__":
        data = DataLoader()
        print('\nLoading DataSet Done!')

        img_train, labels_train = data.load_training()
        train_img = np.array(img_train)
        train_labels = np.array(labels_train)
        print('\nTraining Data has been Loaded!')

        img_test, labels_test = data.load_testing()
        test_img = np.array(img_test)
        test_labels = np.array(labels_test)
        print('\nTesting Data has been Loaded!')

        # Features
        X = train_img

        # Labels
        y = train_labels

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
        print('\nDataSet has been split into train and Validation set! 10% of data will be used as Validation Set')
