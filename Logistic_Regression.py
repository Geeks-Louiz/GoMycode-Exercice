#Import des bibliotheques
import pandas as pd
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# *****************************************************Début traitement ************************************************************
#                                                       Importing data
data = pd.read_csv('titanic-passengers.csv',encoding='UTF-8',delimiter= ';')

data["Survived"]= data["Survived"].replace('No', 0)
data["Survived"]= data["Survived"].replace('Yes', 1)
#--> j'ai changé la valeur de Survived en 0 et 1 car avec AUC et ROC je n'ai pas pu affiché les accruacy car les valeurs étaient string
data.drop("Cabin",inplace=True,axis=1)
data.dropna(inplace=True)
pd.get_dummies(data["Sex"])
sex = pd.get_dummies(data["Sex"],drop_first=True)
embarked = pd.get_dummies(data["Embarked"],drop_first=True)
pclass = pd.get_dummies(data["Pclass"],drop_first=True)
data = pd.concat([data,pclass,sex,embarked],axis=1)
data.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)
X = data.drop("Survived",axis=1)
y = data["Survived"]

#******************************************Splitting and applying LogisticReg***************************************************
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy={:.2f}".format(logmodel.score(X_test, y_test)))

#****************************************** Use confustion matrix *****************************************************************
Confusion_matrix=pd.crosstab(y_test,predictions,rownames=['Actual'],colnames=['Pred'])
print(Confusion_matrix)
print(classification_report(y_test, predictions))

#********************************** Another validation matrix for classification is ROC / AUC , do your research on them explain them and apply them in our case

pred_prob = logmodel.predict_proba(X_test)
probs = pred_prob[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC Score: %.2f' % auc)


fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


#Explication:
#La courbe ROC  est une visualisation du taux de faux positifs et du taux de vrais positifs.
# elle calcule la surface (area)
