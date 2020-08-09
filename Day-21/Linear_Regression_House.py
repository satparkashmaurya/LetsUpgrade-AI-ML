import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score



#Importing Data
Home_Price=pd.read_excel("Linear Regression.xlsx")

# EDA
Home_Price.hist()
Home_Price.corr()
sns.scatterplot(Home_Price['price'],Home_Price['sqft_living'])
sns.scatterplot(Home_Price['price'],Home_Price['sqft_living'])
Home_Price.boxplot()

#Assigning x (depenedent variable)
x=Home_Price.iloc[:,:1]
 
# Assigning y (independent variables)
y=Home_Price.iloc[:,1:]

#Splitting Data for Training and Test
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

#Alias of Linear Regression
lin_reg=LinearRegression()

#Train the Model
lin_reg.fit(X_train,y_train)

# Coeffcient
lin_reg.coef_

# Intercept
lin_reg.intercept_

#Test the Model
ypred=lin_reg.predict(X_test)

#Root Mean Square
RMSE=np.sqrt(mean_squared_error(y_test,ypred))
print(RMSE)

#R Square
r_square=r2_score(y_test,ypred)
print(r_square)

