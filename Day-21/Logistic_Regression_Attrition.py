#Import Libraries 
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import tree, preprocessing


#Import Data
Attrition_Anal =pd.read_csv("general_data.csv")

#Drop Non Required Columns
Attrition_Anal.drop(columns=['EmployeeCount','EmployeeID'],inplace=True)

#Check Null Values
Attrition_Anal.isnull().sum()

# Find the Columns with Null Values
Attrition_Anal.isnull().sum()

# Find the Mean Values to replace
Attrition_Anal.mean()

#Replacing the Null Value  with the Mean Value
#new_Num_Comp_worked =np.where(Attrition_Anal["NumCompaniesWorked"].isnull(),3,Attrition_Anal["NumCompaniesWorked"])
#new_Total_Wrk_Yrs =np.where(Attrition_Anal["TotalWorkingYears"].isnull(),11,Attrition_Anal["TotalWorkingYears"])
#Attrition_Anal["NumCompaniesWorked"]=new_Num_Comp_worked
#Attrition_Anal["TotalWorkingYears"]=new_Total_Wrk_Yrs
Attrition_Anal.interpolate()


#Aliasing Label Encoder
label_encoder =preprocessing.LabelEncoder()

# Encoding
attrition_encoded = Attrition_Anal.apply(label_encoder.fit_transform)

#Define Dependent Variables
Y=attrition_encoded["Attrition"]

#Define Independent Variable
X =attrition_encoded[["Age","BusinessTravel", "Department", "DistanceFromHome", "Education", "EducationField", "Gender", "JobLevel", "JobRole",  "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "Over18", "PercentSalaryHike", "StandardHours", "StockOptionLevel",  "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany","YearsSinceLastPromotion", "YearsWithCurrManager"]]

#Add Constant
X1=sm.add_constant(X)

#Run Logistic Function and Train it
Logistic_Attrition = sm.Logit(Y,X1)
Attr_Result = Logistic_Attrition.fit()

#Error of raise LinAlgError("Singular matrix") means highly correlated features, Identify Those
corr_matrix = attrition_encoded.corr().abs()

#Delete the Columns and reprocess
attrition_encoded.drop(columns=['Over18','StandardHours'],inplace=True)
Y=attrition_encoded["Attrition"]
X =attrition_encoded[["Age","BusinessTravel", "Department", "DistanceFromHome", "Education", "EducationField", "Gender", "JobLevel", "JobRole",  "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike", "StockOptionLevel",  "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany","YearsSinceLastPromotion", "YearsWithCurrManager"]]
X1=sm.add_constant(X)

Logistic_Attrition = sm.Logit(Y,X1)
Attr_Result = Logistic_Attrition.fit()

#Print Results
Attr_Result.summary()



"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:              Attrition   No. Observations:                 4410
Model:                          Logit   Df Residuals:                     4390
Method:                           MLE   Df Model:                           19
Date:                Sun, 09 Aug 2020   Pseudo R-squ.:                  0.1064
Time:                        21:41:32   Log-Likelihood:                -1740.6
converged:                       True   LL-Null:                       -1947.9
Covariance Type:            nonrobust   LLR p-value:                 4.056e-76
===========================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------------
const                      -0.4116      0.301     -1.369      0.171      -1.001       0.178
Age                        -0.0311      0.007     -4.583      0.000      -0.044      -0.018
BusinessTravel             -0.0181      0.065     -0.277      0.782      -0.146       0.110
Department                 -0.2364      0.081     -2.916      0.004      -0.395      -0.077
DistanceFromHome           -0.0011      0.005     -0.199      0.842      -0.012       0.009
Education                  -0.0628      0.043     -1.477      0.140      -0.146       0.021
EducationField             -0.0995      0.033     -2.990      0.003      -0.165      -0.034
Gender                      0.0748      0.089      0.838      0.402      -0.100       0.250
JobLevel                   -0.0292      0.040     -0.739      0.460      -0.107       0.048
JobRole                     0.0358      0.018      2.005      0.045       0.001       0.071
MaritalStatus               0.5897      0.063      9.346      0.000       0.466       0.713
MonthlyIncome              -0.0002      0.000     -1.366      0.172      -0.000     6.7e-05
NumCompaniesWorked          0.0859      0.016      5.379      0.000       0.055       0.117
PercentSalaryHike           0.0134      0.012      1.136      0.256      -0.010       0.036
StockOptionLevel           -0.0636      0.052     -1.233      0.218      -0.165       0.038
TotalWorkingYears          -0.0444      0.011     -3.981      0.000      -0.066      -0.023
TrainingTimesLastYear      -0.1462      0.035     -4.164      0.000      -0.215      -0.077
YearsAtCompany             -0.0039      0.018     -0.213      0.832      -0.040       0.032
YearsSinceLastPromotion     0.1345      0.020      6.574      0.000       0.094       0.175
YearsWithCurrManager       -0.1354      0.022     -6.070      0.000      -0.179      -0.092
===========================================================================================
"""

"""As per above result the Independent Variables that are important are  as p value is less than 0.05:
    Age
    Department
    EducationField
    JobRole
    MaritalStatus
    NumCompaniesWorked
    TotalWorkingYears
    TrainingTimesLastYear
    YearsSinceLastPromotion
    YearsWithCurrManager"""