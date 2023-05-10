import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def split_data(df2,y_training):
  """Split data into features and targets training and test set.
  Args:
    data;data containing features and targets
  returns:
    split data
 """
  X_train,X_test,y_train,y_test=train_test_split(df2,y_training,random_state=0,test_size=0.20)

  return X_train,X_test,y_train,y_test 

def train_model(X_train,X_test,y_train,y_test):
  """Trains the logistic regression model.
  Args:
      X_train:Training data of independent features
      y_train:Training data for price
  Returns:
      Trined model.
 """
  #instantiate the model
  logreg=LogisticRegression(solver='liblinear',random_state=0)

  #fit the model
  logreg = logreg.fit(X_train,y_train)
  return logreg

def evaluate_model(logreg:LogisticRegression, X_test:pd.DataFrame, y_test:pd.Series):
  """calculate and logs the coefficient of determination.
  Args:
      logreg:Trined model
      X_test:Testing data of independent features
      y_test:Testing data for price
  Returns:
      y_pred_test:prediction on x test
      acc        :accuracy_score
  """
  y_pred_test=logreg.predict(X_test)
  #Check accuracy score
  acc=(accuracy_score(y_test,y_pred_test))
  print('model accuracy_score: ',acc)
  return y_pred_test,acc