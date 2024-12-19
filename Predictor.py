from tkinter import *
import tkinter
from tkinter import messagebox
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# load dataset
# print 5 rows of the data frame

CustomersDataset = pd.read_csv("CustomersDataset.csv")
# print(CustomersDataset.head())

# number of rows and columns
# print(CustomersDataset.shape)


# Handel the Data or Cleaning The Data
# ---------------------------------------------------
# print(CustomersDataset.isnull().sum())
# print(CustomersDataset.duplicated())

CustomersDataset.drop_duplicates(inplace=True)

# replacing strings by numeric values to make sense
CustomersDataset["Churn"] = CustomersDataset["Churn"].replace(['No', 'Yes'], [
                                                              0, 1])
CustomersDataset["TotalCharges"] = CustomersDataset["TotalCharges"].replace([
                                                                            ' '], [0])
CustomersDataset["PaperlessBilling"] = CustomersDataset["PaperlessBilling"].replace([
                                                                                    'No', 'Yes'], [0, 1])
CustomersDataset["StreamingMovies"] = CustomersDataset["StreamingMovies"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["StreamingTV"] = CustomersDataset["StreamingTV"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["TechSupport"] = CustomersDataset["TechSupport"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["DeviceProtection"] = CustomersDataset["DeviceProtection"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["OnlineBackup"] = CustomersDataset["OnlineBackup"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["OnlineSecurity"] = CustomersDataset["OnlineSecurity"].replace(
    ['No', 'Yes', 'No internet service'], [0, 1, 2])
CustomersDataset["MultipleLines"] = CustomersDataset["MultipleLines"].replace(
    ['No', 'Yes', 'No phone service'], [0, 1, 2])
CustomersDataset["PhoneService"] = CustomersDataset["PhoneService"].replace([
                                                                            'No', 'Yes'], [0, 1])
CustomersDataset["Dependents"] = CustomersDataset["Dependents"].replace([
                                                                        'No', 'Yes'], [0, 1])
CustomersDataset["Partner"] = CustomersDataset["Partner"].replace(
    ['No', 'Yes'], [0, 1])
CustomersDataset["gender"] = CustomersDataset["gender"].replace(
    ['Female', 'Male'], [0, 1])
CustomersDataset["Contract"] = CustomersDataset["Contract"].replace(
    ['Month-to-month', 'One year', 'Two year'], [0, 1, 2])
CustomersDataset["InternetService"] = CustomersDataset["InternetService"].replace(
    ['DSL', 'Fiber optic', 'No'], [0, 1, 2])

# print(CustomersDataset.head())

# CustomersDataset.corr()

# Drop any features that will not affect our prediction
CustomersDataset.drop(['customerID', 'PaymentMethod'], axis=1, inplace=True)


# to avoid any OverFlow
scaler = MinMaxScaler()
CustomersDataset = pd.DataFrame(scaler.fit_transform(
    CustomersDataset), columns=CustomersDataset.columns)
# print(CustomersDataset.head())

# Split The Data to Train & Test
# --------------------------------
x = CustomersDataset.drop(["Churn"], axis=1)
y = CustomersDataset["Churn"]
# sns.countplot(CustomersDataset["Churn"])
# plt.show()


# We are going to use *SMOTE* tobalance class distribution by randomlyincreasing this minority class examples By Replicating them that's all so now let's use this smooth so for that we
# python
# ```
# from imblearn.over_sampling import SMOTE
# x_res,y_res =SMOTE().fit_resample(x,y)
# y_res.value_counts()
# y_res.value_counts()
# sns.countplot(y_res)
# ```

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=44, shuffle=True)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# print(f"Train size {x_train.shape[0]} rows\nTest size {x_test.shape[0]} rows")
# the shape of train and test y
# print("Shape of Y train: " + str(y_train.shape))
# print("Shape of Y test: " + str(y_test.shape))

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# reshaping it to become a matrix
# print("Shape of Y train: " + str(y_train.shape))
# print("Shape of Y test: " + str(y_test.shape))


# Start Predict Using Logistic Regression
# _______________________________________________________________________________


LR_model = LogisticRegression(
    max_iter=1000, random_state=33, solver='saga', C=1.0)
LR_model.fit(x_train, y_train)
y_pred = LR_model.predict(x_test)
y_pred_prob = LR_model.predict_proba(x_test)
# print(y_pred)

# Calculateing Accuracy score , Precision score , Recall score and F1_Score
# Score for train and test Data , confusing matrix , model classes and number of iterators are used
# _____________________________________________________________________________________________________________________

# print("LogisticRegression_Model Train Score\Accuracy =",
#       LR_model.score(x_train, y_train))
LR_model_score_val= LR_model.score(x_test, y_test)
# print("The confusion matrix For \"Test\" LogisticRegression_Model\n",
#       confusion_matrix(y_test, y_pred))
# print("LogisticRegression_Model Classes :: ", LR_model.classes_)
# print("LogisticRegression_Model Number of Iteratios", LR_model.n_iter_)


# print(classification_report(y_test, y_pred))

# print("The Accuracy Score For \"Test\" LogisticRegression_Model",
#       accuracy_score(y_test, y_pred))
# print("The Precision Score For \"Test\" LogisticRegression_Model",
#       precision_score(y_test, y_pred, average='micro'))
# print("The Recall Score For \"Test\" LogisticRegression_Model",
#       recall_score(y_test, y_pred, average='micro'))
# print("The F1_Score For \"Test\" LogisticRegression_Model",
#       f1_score(y_test, y_pred, average='micro'))


# sns.heatmap(confusion_matrix(y_test, y_pred), center=True)
# plt.show()


# print(" the predict of  10 Rows from Our DataSets = ", y_pred[:10])
# print("the peobabilty of the preidect for 10 Rows From Our DataSets is : \n",
#       y_pred_prob[:10])


# Start Predict Using SVM
# ________________________________________________________________________________________

svm = svm.SVC()
svm.fit(x_train, y_train)
y_pred_2 = svm.predict(x_test)

# Calculateing Accuracy score , Precision score , Recall score and F1_Score
# Score for train and test Data , confusing matrix , model classes and number of iterators are used
# ____________________________________________________________________________________________________________

# print("LogisticRegression_Model Train Score\Accuracy =",
#       svm.score(x_train, y_train))
svm_score_val = svm.score(x_test, y_test)
# print("The confusion matrix For \"Test\" LogisticRegression_Model\n",
#       confusion_matrix(y_test, y_pred))
# print("LogisticRegression_Model Classes :: ", svm.classes_)
# print("LogisticRegression_Model Number of Iteratios", svm.n_iter_)
# print(classification_report(y_test, y_pred_2))

# print("The Accuracy Score For \"Test\" LogisticRegression_Model",
#       accuracy_score(y_test, y_pred_2))
# print("The Precision Score For \"Test\" LogisticRegression_Model",
#       precision_score(y_test, y_pred_2, average='micro'))
# print("The Recall Score For \"Test\" LogisticRegression_Model",
#       recall_score(y_test, y_pred_2, average='micro'))
# print("The F1_Score For \"Test\" LogisticRegression_Model",
#       f1_score(y_test, y_pred_2, average='micro'))

# sns.heatmap(confusion_matrix(y_test, y_pred_2), center=True)
# plt.show()

# print(" the predict of  10 Rows from Our DataSets = ", y_pred[:10])
# print("the peobabilty of the preidect for 10 Rows From Our DataSets is :\n",
#       y_pred_prob[:10])


# Start Predict Using Decision Tree Classifier
# _____________________________________________________________________________________

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_3 = dt.predict(x_test)

accuracy_score(y_test, y_pred_3)

# Calculateing Accuracy score , Precision score , Recall score and F1_Score
# Score for train and test Data , confusing matrix , model classes and number of iterators are used

# print(classification_report(y_test, y_pred_3))

# print("LogisticRegression_Model Train Score\Accuracy =",
#       dt.score(x_train, y_train))
dt_score_val = dt.score(x_test, y_test)
# print("The confusion matrix For \"Test\" LogisticRegression_Model\n",
#       confusion_matrix(y_test, y_pred))
# print("LogisticRegression_Model Classes :: ", dt.classes_)


# print("The Accuracy Score For \"Test\" LogisticRegression_Model",
#       accuracy_score(y_test, y_pred_3))
# print("The Precision Score For \"Test\" LogisticRegression_Model",
#       precision_score(y_test, y_pred_3, average='micro'))
# print("The Recall Score For \"Test\" LogisticRegression_Model",
#       recall_score(y_test, y_pred_3, average='micro'))
# print("The F1_Score For \"Test\" LogisticRegression_Model",
#       f1_score(y_test, y_pred_3, average='micro'))

# sns.heatmap(confusion_matrix(y_test, y_pred_3), center=True)
# plt.show()


# Finally Collecting the Data in DataFrame
# ___________________________________________________________________________________________________________


final_data_Accuracy = pd.DataFrame({'Models': ['LR', 'SVM', 'DT'],
                                    'Accuracy': [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_2), accuracy_score(y_test, y_pred_3)]})

# sns.barplot(final_data_Accuracy['Models'], final_data_Accuracy['Accuracy'])
# plt.show()

final_data_Precision = pd.DataFrame({'Models': ['LR', 'SVM', 'ID3'],
                                     'PRECISION': [precision_score(y_test, y_pred), precision_score(y_test, y_pred_2), precision_score(y_test, y_pred_3)]})


# sns.barplot(final_data_Precision['Models'], final_data_Precision['PRECISION'])
# plt.show()


final_data_Recall = pd.DataFrame({'Models': ['LR', 'SVM', 'ID3'],
                                  'Recall': [recall_score(y_test, y_pred), recall_score(y_test, y_pred_2), recall_score(y_test, y_pred_3)]})

# sns.barplot(final_data_Recall['Models'], final_data_Recall['Recall'])
# plt.show()


final_data_F1_Score = pd.DataFrame({'Models': ['LR', 'SVM', 'ID3'],
                                    'F1_Score': [f1_score(y_test, y_pred), f1_score(y_test, y_pred_2), f1_score(y_test, y_pred_3)]})


# sns.barplot(final_data_F1_Score['Models'], final_data_F1_Score['F1_Score'])
# plt.show()


# Save Our Model
# ______________________________________________________________________


lr_model = joblib.dump(LR_model, 'Churn Predict Model')
svm_model = joblib.dump(svm, 'Churn Predict Model')
dt_model = joblib.dump(dt, 'Churn Predict Model')

lr_model = joblib.load('Churn Predict Model')
svm_model = joblib.load('Churn Predict Model')
dt_model = joblib.load('Churn Predict Model')

# Predict Using Our Model
# ____________________________________________________________________________


# print('Predict Using Logistic Regression', lr_model.predict(
#     [[0.0, 0.0, 1.0, 0.0, 1, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 29.85, 29.85]]))
# print('Predict Using SVM', svm_model.predict(
#     [[0.0, 0.0, 1.0, 0.0, 1, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 29.85, 29.85]]))
# print('Predict Using Decision Tree Classifier', dt_model.predict(
#     [[0.0, 0.0, 1.0, 0.0, 1, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 29.85, 29.85]]))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GUI Using Tkinter
# ___________________________________________________________________________

from tkinter import *

app=Tk()

# Function that used to disply the data we need 
# Data Visualization for our data
def Visualization(): 

    top = Toplevel()
    top.geometry("500x500")
    top.title("    Data Visualization  ")
    
    def dis_1 ():
        sns.countplot(CustomersDataset["Churn"])
        plt.show()

    def dis_2 ():
        sns.heatmap(confusion_matrix(y_test, y_pred), center=True)
        plt.show()

    def dis_3 ():
        sns.heatmap(confusion_matrix(y_test, y_pred_2), center=True)
        plt.show()

    def dis_4 ():
        sns.heatmap(confusion_matrix(y_test, y_pred_3), center=True)
        plt.show()


    def dis_5 ():
        sns.barplot(final_data_Accuracy['Models'], final_data_Accuracy['Accuracy'])
        plt.show()


    def dis_6 ():
        sns.barplot(final_data_Precision['Models'], final_data_Precision['PRECISION'])
        plt.show()

    def dis_7 ():
        sns.barplot(final_data_Recall['Models'], final_data_Recall['Recall'])
        plt.show()

    def dis_8 ():
        sns.barplot(final_data_F1_Score['Models'], final_data_F1_Score['F1_Score'])
        plt.show()

    Button(top,text="CustomersDataset[Churn]",padx=20,pady=10,borderwidth=2.5,command=dis_1).place(x=50,y=50)
    Button(top,text="Confusion Matrix of\n Logistic Regression",padx=20,pady=10,borderwidth=2.5,command=dis_2).place(x=250,y=50)
    Button(top,text="Confusion Matrix of\n     SVM",padx=20,pady=10,borderwidth=2.5,command=dis_3).place(x=50,y=150)
    Button(top,text="Confusion Matrix of\n Decision Tree Classifier",padx=20,pady=10,borderwidth=2.5,command=dis_4).place(x=250,y=150)
    Button(top,text="Final Data Accuracy",padx=20,pady=10,borderwidth=2.5,command=dis_5).place(x=50,y=250)
    Button(top,text="Final Data Precision",padx=20,pady=10,borderwidth=2.5,command=dis_6).place(x=250,y=250)
    Button(top,text="Final Data Recall",padx=20,pady=10,borderwidth=2.5,command=dis_7).place(x=50,y=350)
    Button(top,text="Final Data F1_Score",padx=20,pady=10,borderwidth=2.5,command=dis_8).place(x=250,y=350)

# Rest make all Input Fields are clear to input new data
def Rest():
    E_Customer_ID.delete(0,END)
    E_Gender.delete(0,END)
    E_Senior_Citizen.delete(0,END)
    E_Partner.delete(0,END)
    E_Dependents.delete(0,END)
    E_Tenure.delete(0,END)
    E_Phone_Service.delete(0,END)
    E_Multiple_Lines.delete(0,END)
    E_Internet_Service.delete(0,END)
    E_Online_Security.delete(0,END)
    E_Online_Backup.delete(0,END)
    E_Device_Protection.delete(0,END)
    E_TechSupport.delete(0,END)
    E_Streaming_TV.delete(0,END)
    E_Streaming_Movies.delete(0,END)
    E_Contract.delete(0,END)
    E_Paperless_Billing.delete(0,END)
    E_Monthly_Charges.delete(0,END)
    E_Total_Charges.delete(0,END)
    E_Payment_Method.delete(0,END)

def Display_Train():
    messagebox.showinfo("    Data are relative to Train",f"LogisticRegression_Model Train\n    Score\Accuracy = {LR_model_score_val} \n\nSVM_Model Train \n    Score\Accuracy = {svm_score_val} \n\nDecision_Tree_Model Train  \n   Score\Accuracy = {dt_score_val}\n\n")
    # print("LogisticRegression_Model Train Score\Accuracy =", LR_model.score(x_train,y_train))
    # print("SVM_Model Train Score\Accuracy =", svm.score(x_train,y_train))
    # print("Decision_Tree_Model Train Score\Accuracy =", dt.score(x_train,y_train))

def Display_Test():
    messagebox.showinfo("    Data are relative to Test",f"Churn Predict Model Accuracy Score\n{final_data_Accuracy}\n\nChurn Predict Model Precision Score\n{final_data_Precision}\n\nChurn Predict Model Recall Score\n{final_data_Recall}\n\n Churn Predict Model F1_Score\n{final_data_F1_Score}\n\nThe confusion matrix For \"Test\" LogisticRegression_Model\n{confusion_matrix(y_test,y_pred)}\n\nThe confusion matrix For \"Test\" SVM_Model\n{confusion_matrix(y_test,y_pred_2)}\n\nThe confusion matrix For \"Test\" DecisionTreeClassifier_Model\n{confusion_matrix(y_test,y_pred_3)}\n\n")
    # print('***************************************************************************************************************************************')
    # print("Churn Predict Model Accuracy Score\n",final_data_Accuracy)
    # print("Churn Predict Model Precision Score\n",final_data_Precision)
    # print("Churn Predict Model Recall Score\n",final_data_Recall)
    # print("Churn Predict Model F1_Score\n",final_data_F1_Score)
    # print("The confusion matrix For \"Test\" LogisticRegression_Model\n",confusion_matrix(y_test,y_pred))
    # print("The confusion matrix For \"Test\" SVM_Model\n",confusion_matrix(y_test,y_pred_2))
    # print("The confusion matrix For \"Test\" DecisionTreeClassifier_Model\n",confusion_matrix(y_test,y_pred_3))
    # print('********************************************************************************************************************')


def Predict_Ndata ():
    global E_Gender_val
    global E_Senior_Citizen_val
    global E_Partner_val
    global E_Dependents_val
    global E_Tenure_val

    global E_Phone_Service_val
    global E_Multiple_Lines_val
    global E_Internet_Service_val
    global E_Online_Security_val

    global E_Online_Backup_val
    global E_Device_Protection_val
    global E_TechSupport_val
    global E_Streaming_TV_val

    global E_Streaming_Movies_val
    global E_Contract_val
    global E_Paperless_Billing_val
    global E_Monthly_Charges_val
    global E_Total_Charges_val

    E_Monthly_Charges_val =E_Monthly_Charges .get()
    E_Total_Charges_val =E_Total_Charges.get()
    E_Tenure_val = E_Tenure.get()

    # E_Partner
    if E_Partner.get()=="no" or E_Partner.get()=="No" or E_Partner.get()=="NO":
        E_Partner_val =0
    if E_Partner.get()=="yes" or E_Partner.get()=="Yes" or E_Partner.get()=="YES":
        E_Partner_val =1 #############

    # E_Dependents
    if E_Dependents.get()=="no" or E_Dependents.get()=="No" or E_Dependents.get()=="NO":
        E_Dependents_val =0
    if E_Dependents.get()=="yes" or E_Dependents.get()=="Yes" or E_Dependents.get()=="YES":
        E_Dependents_val =1 #############

    # E_Phone_Service
    if E_Phone_Service.get()=="no" or E_Phone_Service.get()=="No" or E_Phone_Service.get()=="NO":
        E_Phone_Service_val =0
    if E_Phone_Service.get()=="yes" or E_Phone_Service.get()=="Yes" or E_Phone_Service.get()=="YES":
        E_Phone_Service_val =1 #############

    # E_Paperless_Billing
    if E_Paperless_Billing.get()=="no" or E_Paperless_Billing.get()=="No" or E_Paperless_Billing.get()=="NO":
        E_Paperless_Billing_val =0
    if E_Paperless_Billing.get()=="yes" or E_Paperless_Billing.get()=="Yes" or E_Paperless_Billing.get()=="YES":
        E_Paperless_Billing_val =1 #############

    # E_Senior_Citizen
    if E_Senior_Citizen.get()=="no" or E_Senior_Citizen.get()=="No" or E_Senior_Citizen.get()=="NO":
        E_Senior_Citizen_val =0
    if E_Senior_Citizen.get()=="yes" or E_Senior_Citizen.get()=="Yes" or E_Senior_Citizen.get()=="YES":
        E_Senior_Citizen_val =1 #############


    # E_Gender
    if E_Gender.get()=="Female" or E_Gender.get()=="female" or E_Gender.get()=="FEMALE":
        E_Gender_val =0
    if E_Gender.get()=="Male" or E_Gender.get()=="male" or E_Gender.get()=="MALE":
        E_Gender_val =1 #############


    # E_Multiple_Lines
    if E_Multiple_Lines.get()=="no" or E_Multiple_Lines.get()=="No" or E_Multiple_Lines.get()=="NO":
        E_Multiple_Lines_val =0
    if E_Multiple_Lines.get()=="yes" or E_Multiple_Lines.get()=="Yes" or E_Multiple_Lines.get()=="YES":
        E_Multiple_Lines_val =1
    if E_Multiple_Lines.get()=="No phone service" or E_Multiple_Lines.get()=="no phone service" :
        E_Multiple_Lines_val =2 #############

    # E_Internet_Service
    if E_Internet_Service.get()=="DSL" or E_Internet_Service.get()=="dsl":
        E_Internet_Service_val =0
    if E_Internet_Service.get()=="Fiber obtic" or E_Internet_Service.get()=="FIBER OBTIC" or E_Internet_Service.get()=="Fiber Obtic":
        E_Internet_Service_val =1
    if E_Internet_Service.get()=="No" or E_Multiple_Lines.get()=="no" or E_Multiple_Lines.get()=="NO" :
        E_Internet_Service_val =2 #############

    # E_Online_Security
    if E_Online_Security.get()=="no" or E_Online_Security.get()=="No" or E_Online_Security.get()=="NO":
        E_Online_Security_val =0
    if E_Online_Security.get()=="yes" or E_Online_Security.get()=="Yes" or E_Online_Security.get()=="YES":
        E_Online_Security_val =1
    if E_Online_Security.get()=="No internet service" or E_Online_Security.get()=="no internet service" :
        E_Online_Security_val =2 #############

    # E_Online_Backup    
    if E_Online_Backup.get()=="no" or E_Online_Backup.get()=="No" or E_Online_Backup.get()=="NO":
        E_Online_Backup_val =0
    if E_Online_Backup.get()=="yes" or E_Online_Backup.get()=="Yes" or E_Online_Backup.get()=="YES":
        E_Online_Backup_val =1
    if E_Online_Backup.get()=="No internet service" or E_Online_Backup.get()=="no internet service" :
        E_Online_Backup_val =2 #############

    # E_Device_Protection
    if E_Device_Protection.get()=="no" or E_Device_Protection.get()=="No" or E_Device_Protection.get()=="NO":
        E_Device_Protection_val =0
    if E_Device_Protection.get()=="yes" or E_Device_Protection.get()=="Yes" or E_Device_Protection.get()=="YES":
        E_Device_Protection_val =1
    if E_Device_Protection.get()=="No internet service" or E_Device_Protection.get()=="no internet service" :
        E_Device_Protection_val =2 #############

    # E_TechSupport
    if E_TechSupport.get()=="no" or E_TechSupport.get()=="No" or E_TechSupport.get()=="NO":
        E_TechSupport_val =0
    if E_TechSupport.get()=="yes" or E_TechSupport.get()=="Yes" or E_TechSupport.get()=="YES":
        E_TechSupport_val =1
    if E_TechSupport.get()=="No internet service" or E_TechSupport.get()=="no internet service" :
        E_TechSupport_val =2 #############

    # E_Streaming_TV
    if E_Streaming_TV.get()=="no" or E_Streaming_TV.get()=="No" or E_Streaming_TV.get()=="NO":
        E_Streaming_TV_val =0
    if E_Streaming_TV.get()=="yes" or E_Streaming_TV.get()=="Yes" or E_Streaming_TV.get()=="YES":
        E_Streaming_TV_val =1
    if E_Streaming_TV.get()=="No internet service" or E_Streaming_TV.get()=="no internet service" :
        E_Streaming_TV_val =2 #############

    # E_Streaming_Movies
    if E_Streaming_Movies.get()=="no" or E_Streaming_Movies.get()=="No" or E_Streaming_Movies.get()=="NO":
        E_Streaming_Movies_val =0
    if E_Streaming_Movies.get()=="yes" or E_Streaming_Movies.get()=="Yes" or E_Streaming_Movies.get()=="YES":
        E_Streaming_Movies_val =1
    if E_Streaming_Movies.get()=="No internet service" or E_Streaming_Movies.get()=="no internet service" :
        E_Streaming_Movies_val =2 #############

    # E_Contract   
    if E_Contract.get()=="Month-to-month" or E_Contract.get()=="month-to-month" or E_Contract.get()=="month to month" or E_Contract.get()=="Month to month":
        E_Contract_val =0
    if E_Contract.get()=="One year" or E_Contract.get()=="one year":
        E_Contract_val =1
    if E_Contract.get()=="Two year" or E_Contract.get()=="two year" :
        E_Contract_val =2 #############

    # Message Box For our Prediction 
    lr_model_val = lr_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]])
    svm_model_val = svm_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]])
    dt_model_val = dt_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]])
    
    global Churn_Predict_LR
    global Churn_Predict_SNM
    global Churn_Predict_DT

    if lr_model_val==1 :
        Churn_Predict_LR = "Yes "
    if lr_model_val==0 :
        Churn_Predict_LR = "No "

        #######################

    if svm_model_val==1 :
        Churn_Predict_SNM = "Yes "
    if svm_model_val==0 :
        Churn_Predict_SNM = "No "

        ###########################

    if dt_model_val==1 :
        Churn_Predict_DT = "Yes "
    if dt_model_val==0 :
        Churn_Predict_DT = "No "
    
    
    
    messagebox.showinfo("  Prediction  ",f"Churn Predict Using Logistic Regression =  {Churn_Predict_LR}\nChurn Predict Using SVM =  {Churn_Predict_SNM} \nChurn Predict Using Decision Tree Classifier  =  {Churn_Predict_DT}")

    # else:
    #     messagebox.showerror("    ERORR!!!!","\n    INVALED DATA \n ")





    # print('Predict Using Logistic Regression',lr_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]]))
    # print('Predict Using SVM',svm_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]]))
    # print('Predict Using Decision Tree Classifier',dt_model.predict([[E_Gender_val,E_Senior_Citizen_val,E_Partner_val,E_Dependents_val,E_Tenure_val,E_Phone_Service_val,E_Multiple_Lines_val,E_Internet_Service_val,E_Online_Security_val,E_Online_Backup_val,E_Device_Protection_val,E_TechSupport_val,E_Streaming_TV_val,E_Streaming_Movies_val,E_Contract_val,E_Paperless_Billing_val,E_Monthly_Charges_val,E_Total_Charges_val]]))

    # Index(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    #    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    #    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    #    'StreamingMovies', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
    #    'TotalCharges'],
    # E_Customer_ID
    # E_Payment_Method





app.title("Churn Predict Model")
app.geometry("1000x800")
# methodology 2 Labels
Label(app,text='              Mehodology', font=("Arial", 12)).place(x=130,y=4)
Label(app,text='               Logistic Regression   -  SVM  -   Decision Tree Classifier', font=("Arial", 12)).place(x=10,y=30)
# 2 Buttons to display all data you have for Train & Test
Train_btn=Button(app, text='Train', padx=40, pady=10,borderwidth=2.5,command=Display_Train).place(x=230,y=70)
Test_btn=Button(app, text='Test', padx=40, pady=10,borderwidth=2.5,command=Display_Test).place(x=350,y=70)
# Customer Data 20 Labels & 20 Input Fields
Label(app,text='Customer Data', font=("Arial", 10)).place(x=60,y=115)
# 20 Labels start Using Place
Label(app,text='Customer ID', font=("Arial", 10)).place(x=20,y=180)
Label(app,text='Partner', font=("Arial", 10)).place(x=20,y=240)
Label(app,text='Phone Service', font=("Arial", 10)).place(x=20,y=300)
Label(app,text='Online Security', font=("Arial", 10)).place(x=20,y=360)
Label(app,text='TechSupport', font=("Arial", 10)).place(x=20,y=420)
Label(app,text='Contract', font=("Arial", 10)).place(x=20,y=480)
Label(app,text='Monthly Charges', font=("Arial", 10)).place(x=20,y=540)

Label(app,text='Gender', font=("Arial", 10)).place(x=300,y=180)
Label(app,text='Dependents', font=("Arial", 10)).place(x=300,y=240)
Label(app,text='Multiple Lines', font=("Arial", 10)).place(x=300,y=300)
Label(app,text='Online Backup', font=("Arial", 10)).place(x=300,y=360)
Label(app,text='Streaming TV', font=("Arial", 10)).place(x=300,y=420)
Label(app,text='Paperless Billing', font=("Arial", 10)).place(x=300,y=480)
Label(app,text='Total Charges', font=("Arial", 10)).place(x=300,y=540)

Label(app,text='Senior Citizen', font=("Arial", 10)).place(x=580,y=180)
Label(app,text='Tenure', font=("Arial", 10)).place(x=580,y=240)
Label(app,text='Internet Service', font=("Arial", 10)).place(x=580,y=300)
Label(app,text='Device Protection', font=("Arial", 10)).place(x=580,y=360)
Label(app,text='Streaming Movies', font=("Arial", 10)).place(x=580,y=420)
Label(app,text='Payment Method', font=("Arial", 10)).place(x=580,y=480)


# 20 Input Fields to take data from user 
E_Customer_ID=Entry(app,width=20,borderwidth=2.5)
E_Customer_ID.place(x=135,y=180)

E_Partner=Entry(app,width=20,borderwidth=2.5)
E_Partner.place(x=135,y=240)
E_Phone_Service=Entry(app,width=20,borderwidth=2.5)
E_Phone_Service.place(x=135,y=300)
E_Online_Security=Entry(app,width=20,borderwidth=2.5)
E_Online_Security.place(x=135,y=360)
E_TechSupport=Entry(app,width=20,borderwidth=2.5)
E_TechSupport.place(x=135,y=420)
E_Contract=Entry(app,width=20,borderwidth=2.5)
E_Contract.place(x=135,y=480)
E_Monthly_Charges=Entry(app,width=20,borderwidth=2.5)
E_Monthly_Charges.place(x=135,y=540)


E_Gender=Entry(app,width=20,borderwidth=2.5)
E_Gender.place(x=415,y=180)
E_Dependents=Entry(app,width=20,borderwidth=2.5)
E_Dependents.place(x=415,y=240)
E_Multiple_Lines=Entry(app,width=20,borderwidth=2.5)
E_Multiple_Lines.place(x=415,y=300)
E_Online_Backup=Entry(app,width=20,borderwidth=2.5)
E_Online_Backup.place(x=415,y=360)
E_Streaming_TV=Entry(app,width=20,borderwidth=2.5)
E_Streaming_TV.place(x=415,y=420)
E_Paperless_Billing=Entry(app,width=20,borderwidth=2.5)
E_Paperless_Billing.place(x=415,y=480)
E_Total_Charges=Entry(app,width=20,borderwidth=2.5)
E_Total_Charges.place(x=415,y=540)


E_Senior_Citizen=Entry(app,width=20,borderwidth=2.5)
E_Senior_Citizen.place(x=695,y=180)
E_Tenure=Entry(app,width=20,borderwidth=2.5)
E_Tenure.place(x=695,y=240)
E_Internet_Service=Entry(app,width=20,borderwidth=2.5)
E_Internet_Service.place(x=695,y=300)
E_Device_Protection=Entry(app,width=20,borderwidth=2.5)
E_Device_Protection.place(x=695,y=360)
E_Streaming_Movies=Entry(app,width=20,borderwidth=2.5)
E_Streaming_Movies.place(x=695,y=420)
E_Payment_Method=Entry(app,width=20,borderwidth=2.5)
E_Payment_Method.place(x=695,y=480)

# Btn to predict using the data user enterd , Btn to clear Input Field to enter new data
Predict_btn= Button(app,text='Predict',padx=60,pady=20,borderwidth=2.5,command=Predict_Ndata).place(x=190,y=600)###########command###########
Clear_btn= Button(app,text='Rest',padx=60,pady=20,borderwidth=2.5,command=Rest).place(x=372,y=600)###########command###########
Data_Visualization= Button(app,text='Data Visualization',padx=60,pady=20,borderwidth=2.5,command=Visualization).place(x=540,y=600)###########command#################################################################










app.mainloop() 
















