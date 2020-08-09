#Importing  Required Libraires
import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from sklearn.ensemble import RandomForestClassifier

#Creating Data Frame with Required Information
attrition_train =pd.read_csv("G:/AIML/Assignments-Received/Day-24/Dataset/general_data.csv")

#Drop Non Relevant Columns
attrition_train.drop(columns=['EmployeeCount','EmployeeID'],inplace=True)


# Find the Columns with Null Values
attrition_train.isnull().sum()

# Find the Mean Values to replace
attrition_train.mean()



#Replacing the Null Value  with the Mean Value
new_Num_Comp_worked =np.where(attrition_train["NumCompaniesWorked"].isnull(),3,attrition_train["NumCompaniesWorked"])
new_Total_Wrk_Yrs =np.where(attrition_train["TotalWorkingYears"].isnull(),11,attrition_train["TotalWorkingYears"])
attrition_train["NumCompaniesWorked"]=new_Num_Comp_worked
attrition_train["TotalWorkingYears"]=new_Total_Wrk_Yrs

#finding categories to convert
objList = attrition_train.select_dtypes(include = "object").columns

#Converting Catgories
label_encoder =preprocessing.LabelEncoder()
attrition_encoded = attrition_train.apply(label_encoder.fit_transform)

# Definig Random Forest Model
rf_att_model = RandomForestClassifier(n_estimators=1000,max_features =2,oob_score=True)

#Defining Features to Analyse
features =["Age","BusinessTravel", "Department", "DistanceFromHome", "Education", "EducationField", "Gender", "JobLevel", "JobRole",  "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "Over18", "PercentSalaryHike", "StandardHours", "StockOptionLevel",  "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany","YearsSinceLastPromotion", "YearsWithCurrManager"]

#Training / Fitting in Random Forest Model
rf_att_model.fit(X=attrition_encoded[features],y=attrition_encoded["Attrition"])
 
#Checking the OOB(out of Bag) Score for identifying  important Independent Variable
print("OOB Accuracy :",rf_att_model.oob_score_)


# Finding importance of the features
for feature, imp in zip(features,rf_att_model.feature_importances_):
    print(feature,imp)
    

#Assigning Variable to the Decision Tress
tree_model =tree.DecisionTreeClassifier()

#Selecting the Independent Variables and passing in Tree 'T'
IndVar_Attr =pd.DataFrame([attrition_encoded['Age'],attrition_encoded['MonthlyIncome'],attrition_encoded['TotalWorkingYears'],attrition_encoded['DistanceFromHome'],attrition_encoded['YearsAtCompany'],attrition_encoded['PercentSalaryHike'],attrition_encoded['NumCompaniesWorked'],attrition_encoded['JobRole'],attrition_encoded['YearsWithCurrManager']]).T

#Training the Model
attr_tree_model.fit(X=IndVar_Attr,y=attrition_encoded["Attrition"])

#Storing the Output for visualisation
with open("Attr_Model.dot",'w') as f:
    f=tree.export_graphviz(attr_tree_model,feature_names=["Age","MonthlyIncome","TotalWorkingYears","DistanceFromHome","YearsAtCompany","PercentSalaryHike","NumCompaniesWorked","JobRole","YearsWithCurrManager"],out_file=f);

#Printing the Classification Accuracy
print("Classification Accuracy with the features  is :", attr_tree_model.score(X=IndVar_Attr,y=attrition_encoded["Attrition"])