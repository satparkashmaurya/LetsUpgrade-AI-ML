#Importing  Required Libraires
import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from sklearn.ensemble import RandomForestClassifier

#Creating Data Frame with Required Information
Bank_Loan =pd.read_excel("G:/AIML/Assignments-Received/Day-24/Dataset/Bank_Personal_Loan_Modelling.xlsx",sheet_name="Data")

#Drop Non Relevant Columns
Bank_Loan.drop(columns=['ID','ZIP Code'],inplace=True)

# Find the Columns with Null Values
Bank_Loan.isnull().sum()

#finding categories to convert
objList = Bank_Loan.select_dtypes(include = "object").columns
print(objList)


# Definig Random Forest Model
rf_bank_model = RandomForestClassifier(n_estimators=1000,max_features =2,oob_score=True)

#Defining Features to Analyse
features =["Age", "Experience", "Income", "Family", "CCAvg", "Education","Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"]

#Training / Fitting in Random Forest Model
rf_bank_model.fit(X=Bank_Loan[features],y=Bank_Loan["Personal Loan"])
 
#Checking the OOB(out of Bag) Score for identifying  important Independent Variable
print("OOB Accuracy :",rf_bank_model.oob_score_)


# Finding importance of the features
for feature, imp in zip(features,rf_bank_model.feature_importances_):
    print(feature,imp)
    

#Assigning Variable to the Decision Tress
bank_tree_model =tree.DecisionTreeClassifier()

#Selecting the Independent Variables and passing in Tree 'T'
IndVar_Bank =pd.DataFrame([Bank_Loan["Income"],Bank_Loan["CCAvg"],Bank_Loan["Education"],Bank_Loan["Family"]]).T

#Training the Model
bank_tree_model.fit(X=IndVar_Bank,y=Bank_Loan["Personal Loan"])

#Storing the Output for visualisation
with open("Bank_Model.dot",'w') as f:
    f=tree.export_graphviz(bank_tree_model,feature_names=["Income","CCAvg","Education","Family"],out_file=f);

#Printing the Classification Accuracy
print("Classification Accuracy with the features  is :",bank_tree_model.score(X=IndVar_Bank,y=Bank_Loan["Personal Loan"]))
