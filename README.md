# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```PYTHON
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHISH S
RegisterNumber:  212224240105
import pandas as pd 
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Chiiradeep R
RegisterNumber: 212224240028
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
HEAD:
![image](https://github.com/user-attachments/assets/522e777c-76e8-4410-bff0-dbca12a7069f)

COPY:
![image](https://github.com/user-attachments/assets/51d8273e-4694-412f-b5c6-468e9b9ae4e8)

FIT TRANSFORM:
![image](https://github.com/user-attachments/assets/d2643821-0ad8-499f-98f4-7048d0ead584)

LOGISTIC REGRESSION:
![image](https://github.com/user-attachments/assets/a1a3a017-a03d-40fc-a67e-5224d57636f3)
![image](https://github.com/user-attachments/assets/e14745e0-37d2-4b87-933b-a3d05efe2050)

ACCURACY:
![image](https://github.com/user-attachments/assets/689021bb-5eba-4fbf-9ee0-cba6aebffbb2)
![image](https://github.com/user-attachments/assets/03d3ac56-69d4-4507-a213-1b6f66dd6789)
![image](https://github.com/user-attachments/assets/181091bc-5f43-4efb-be65-a18d3f3fe7b5)

PREDICTION:
![image](https://github.com/user-attachments/assets/ab6b37c3-e335-45fd-ab71-55eae177e047)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
