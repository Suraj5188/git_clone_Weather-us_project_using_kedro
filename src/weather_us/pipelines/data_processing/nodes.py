import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn import preprocessing


def extract_training_data(df):
  """
  Drop date column from dataframe.
  Arg=df
  output=Dataframe with no date column.
  """
  df=df.drop(["Date"],axis=1)
  df1=df[df['RainTomorrow'].notna()] 
  return df1

def treat_missing(df1):
  """
  treat missing values with ffill and bfill method

  Arg=df_training_data

  output=nan values from training data get fill with the ffill and bfill method."""
  df1_treat_training_data=df1.fillna(method="ffill",axis=0).fillna(method="bfill",axis=0)
  return df1_treat_training_data

def training_data_split(df1_treat_training_data):
  """
  Training data spliting ie.separating predictors and response variables.

  Arg=df_treat_training_data

  Output=Target variable get separated from main traing data."""
  X_training=df1_treat_training_data.drop(["RainTomorrow"],axis=1)
  y_training=df1_treat_training_data["RainTomorrow"]
  return X_training, y_training

def lebel_encoding_filling_null(X_training):
  """
  Labal encoding on the discrite varibales.

  Arg=X_training

  Output=converting categorical data into numeric variable."""
  
  label_encoder = preprocessing.LabelEncoder()
  
  X_training["Location"]=label_encoder.fit_transform(X_training["Location"])
  X_training["WindGustDir"]=label_encoder.fit_transform(X_training["WindGustDir"])
  X_training["WindDir9am"]=label_encoder.fit_transform(X_training["WindDir9am"])
  X_training["WindDir3pm"]=label_encoder.fit_transform(X_training["WindDir3pm"])
  X_training["RainToday"]=label_encoder.fit_transform(X_training["RainToday"])
  df2 = X_training.copy()
  return df2