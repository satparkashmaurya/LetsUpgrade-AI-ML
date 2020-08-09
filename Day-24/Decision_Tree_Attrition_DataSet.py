#Importing  Required Libraires
import pandas as pd
import numpy as np
from sklearn import tree, preprocessing

#Creating Data Frame with Required Information
titanic_train =pd.read_csv("G:/AIML/Assignments-Received/Day-24/Dataset/general_data.csv")

#Replacing the Null Value in the Feature Used (Age) with the Mean Value
new_age_var = np.where(titanic_train["Age"].isnull(),32,titanic_train["Age"])
titanic_train["Age"]=new_age_var
label_encoder =preprocessing.LabelEncoder()
encoded_gender = label_encoder.fit_transform(titanic_train["Sex"])

#Assigning Variable to the Decision Tress
tree_model =tree.DecisionTreeClassifier()

#Selecting the Independent Variables and passing in Tree 'T'
predictors = pd.DataFrame([encoded_gender,titanic_train["Age"],titanic_train["Fare"]]).T

#Training the Model
tree_model.fit(X=predictors,y=titanic_train["Survived"])

#Storing the Output for visualisation
with open("TitanicDtree.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Sex","Age","Fare"],out_file=f);

#Printing the Classification Accuracy
print("Classification Accuracy with the features ,Gender, Age & Fare is :", tree_model.score(X=predictors,y=titanic_train["Survived"]))
