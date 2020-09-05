import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.ensemble import RandomForestClassifier
import pydot


#                                                       Importing data
data = pd.read_csv('titanic-passengers.csv',encoding='UTF-8',delimiter= ';')
def preprocess_data(new_data):
    new_data['Age'].fillna(new_data['Age'].mean(),inplace=True)
    new_data.replace({'Sex':{'male': 1,'female':0}},inplace=True)
    new_data['Cabin']=new_data.Cabin.fillna('G6')
    new_data.replace({'Survived':{'Yes': 1,'No':0}},inplace=True)
    return new_data
data_process=preprocess_data(data)

#***************************************************** Decision and Randomforest******************************************

X=data_process.drop(["Survived", "Name", "Cabin", "Ticket", "Embarked"], axis=1)
Y= data_process["Survived"]

#splitting data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20,random_state=10)
all_features=list(X.columns)
#applying tree algorithm
tree_classifier = tree.DecisionTreeClassifier()

tree_classifier.fit(x_train, y_train)   #fitting our model
tree.plot_tree(tree_classifier)
y_pred=tree_classifier.predict(x_test)   # evaluating our model
print("score:{}".format(accuracy_score(y_test, y_pred)))

dot_data = tree.export_graphviz(tree_classifier,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                special_characters=True,
                               feature_names =all_features )
graph=graphviz.Source(dot_data)
graph.render("data")
graph

tree.plot_tree(tree_classifier)

#************************************************************* Change parametres***************************************************
parameters_dt = tree_classifier.get_params()
parameters_dt['max_depth'] = 3
parameters_dt['criterion'] = 'entropy'
new_clf = DecisionTreeClassifier(**parameters_dt)

new_clf.fit(x_train, y_train)   #fitting our model

y_pred=new_clf.predict(x_test)   # evaluating our model
print("score:{}".format(accuracy_score(y_test, y_pred))) # acc = 0.80 en minimisant la profondeur de recherche et en changeant la fonction de gain on a maximiser la précision

#******************************************************* Random Forest**************************************************************

Classifier_For=RandomForestClassifier(n_estimators=10)
Classifier_For.fit(x_train, y_train)
y_pred=Classifier_For.predict(x_test)
print("Accuracy for Random Forest with 10 n_estimators:", accuracy_score(y_test, y_pred))


#****************************************************** Changing n_estimators *****************************************************
New_Classifier_For=RandomForestClassifier(n_estimators=2)
New_Classifier_For.fit(x_train, y_train)
y_pred=New_Classifier_For.predict(x_test)
print("Accuracy for Random Forest with 2 n_estimators:", accuracy_score(y_test, y_pred))

#************************************************** n_estimators******************************************************************
New1_Classifier_For=RandomForestClassifier(n_estimators=20)
New1_Classifier_For.fit(x_train, y_train)
y_pred=New1_Classifier_For.predict(x_test)
print("Accuracy for Random Forest with 20 n_estimators:", accuracy_score(y_test, y_pred))


#** Conclusion :
# Random forest with 2 decision tree gives accruacy= 0.80, with 10 decisio trees gives acc= 0.81, finally when we teste it with 20 we grow the acc to 0.83
#Plus il y'a des arbres de décision dans le random forest plus on utilise les differents subsets of features and differents decision trees et plus on s'éloigne du overfitting


# Le graphe de L'arbre de decison de la question 2 est le fichier data.pdf