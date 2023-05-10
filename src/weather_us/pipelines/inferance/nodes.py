import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



def extracting_inference_data(df):
    """ Extracting inference data
    Arg: df
    return: df_inf"""
    df = df.drop('Date',axis=1)
    df_inference = df[df['RainTomorrow'].isna()]
    return df_inference

def splitting_inference_data(df_inference):
    """ Splitting inference data
    Arg: df2
    Return: X_inf,y_inf """
    X_inference = df_inference.drop(['RainTomorrow'],axis=1)
    y_inference = df_inference['RainTomorrow']
    return X_inference,y_inference

def inference_data_treat_missing_val(X_inference):
    """ Misssing value treatment on training data
    Arg: df1
    Return: df1_treat_missing_value"""
    Xinf_treat_missing_value = X_inference.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)
    return Xinf_treat_missing_value

def inference_data_label_encoding(Xinf_treat_missing_value):
    """ Label encoding converting categorical variables to numerical variables
    Arg: df1_treat_missing_value
    Return: df1_label_encoder"""
    l_encoder = LabelEncoder()
    Xinf_treat_missing_value['Location'] = l_encoder.fit_transform(Xinf_treat_missing_value['Location'])
    Xinf_treat_missing_value['WindGustDir'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindGustDir'])
    Xinf_treat_missing_value['WindDir9am'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindDir9am'])
    Xinf_treat_missing_value['WindDir3pm'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindDir3pm'])
    Xinf_treat_missing_value['RainToday'] = l_encoder.fit_transform(Xinf_treat_missing_value['RainToday'])
    Xinf_treat_missing_values = Xinf_treat_missing_value.copy()
    return Xinf_treat_missing_values

def log_reg_Algorithm(Xinf_treat_missing_values:pd.DataFrame,logreg) -> pd.DataFrame:
  """calculate and logs the coefficient of determination.
  Args:
      logreg:Trined model
      X_test:Testing data of independent features
      y_test:Testing data for price
  Returns:
  """
  y_pred_inference=logreg.predict(Xinf_treat_missing_values)
  data_y_pred_inf = pd.DataFrame(y_pred_inference)
  return data_y_pred_inf
