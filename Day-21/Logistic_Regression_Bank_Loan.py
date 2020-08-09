#Import Libraries 
import pandas as pd
import statsmodels.api as sm

#Import Data
Bank_Loan = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", sheet_name ='Data')

#Drop Non Required Columns
Bank_Loan.drop(columns=['ID','ZIP Code'],inplace=True)

#Check Null Values
Bank_Loan.isnull().sum()

#Define Dependent Variables
Y=Bank_Loan["Personal Loan"]

#Define Independent Variable
X =Bank_Loan[["Age", "Experience", "Income", "Family", "CCAvg", "Education","Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"]]

#Add Constant
X1=sm.add_constant(X)

#Run Logistic Function and Train it
Logistic_Bank = sm.Logit(Y,X1)
Bank_Result = Logistic_Bank.fit()

#Print Results
Bank_Result.summary()

"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:          Personal Loan   No. Observations:                 5000
Model:                          Logit   Df Residuals:                     4988
Method:                           MLE   Df Model:                           11
Date:                Sun, 09 Aug 2020   Pseudo R-squ.:                  0.5938
Time:                        19:52:52   Log-Likelihood:                -642.18
converged:                       True   LL-Null:                       -1581.0
Covariance Type:            nonrobust   LLR p-value:                     0.000
======================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------
const                -12.1928      1.645     -7.411      0.000     -15.417      -8.968
Age                   -0.0536      0.061     -0.874      0.382      -0.174       0.067
Experience             0.0638      0.061      1.046      0.295      -0.056       0.183
Income                 0.0546      0.003     20.831      0.000       0.049       0.060
Family                 0.6958      0.074      9.364      0.000       0.550       0.841
CCAvg                  0.1240      0.040      3.127      0.002       0.046       0.202
Education              1.7362      0.115     15.088      0.000       1.511       1.962
Mortgage               0.0005      0.001      0.856      0.392      -0.001       0.002
Securities Account    -0.9368      0.286     -3.277      0.001      -1.497      -0.377
CD Account             3.8225      0.324     11.800      0.000       3.188       4.457
Online                -0.6752      0.157     -4.298      0.000      -0.983      -0.367
CreditCard            -1.1197      0.205     -5.462      0.000      -1.522      -0.718
======================================================================================

As per above result the Independent Variables that are important are  :- Income,Family,CCAvg,Education,Securities Account,CD Account,Online,CreditCard  

"""