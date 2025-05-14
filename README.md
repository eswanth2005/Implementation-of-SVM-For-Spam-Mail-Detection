# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contests to be displayed using df.head()
3. Split the dataset using train_test_split.
4. Calculate Y_pred and accuracy.
5. Print all the outputs.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ESWANTH KUMAR K
RegisterNumber: 212223040046
*/
```
```
import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'Windows-1252') 
data.head()
data.isnull().sum()
x = data["v1"].values
y = data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
#Countvectorizer is a method to convert text to numerical data.The text is transformed to a sparse data
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```


## Output:

## data.head()
![379537361-793cdbf1-6982-40b6-b579-664f18455db8](https://github.com/user-attachments/assets/4227d664-7cf0-43af-9f2b-14cbbc5b68c4)
## data.info()
![379540277-84e0a36c-cd22-4072-aa6a-88d97401e192](https://github.com/user-attachments/assets/516fc68e-c5db-4a96-b5f1-791a3ec5179f)
## y.pred()
![379539787-39766337-f7d0-4aca-9ae2-09463c744bf7](https://github.com/user-attachments/assets/7cafdeb8-772a-4106-855b-81414e2674af)
## accuracy score
![379539973-ee1c7fa8-f190-4970-8596-dd58e512f78b](https://github.com/user-attachments/assets/84248f33-9e0e-40f3-b344-f86b41a20126)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
