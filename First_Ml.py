# ********************************************* Libraries*****************************************************************#
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
# ********************************************* Loading data **************************************************************#

Data_House = pd.read_csv('kc_house_data.csv',encoding='UTF-8',delimiter= ',')
#print(Data_House)

# ********************************************* Prétraitement *************************************************************#
# Apercu
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#print(Data_House.info())
#print(Data_House.dtypes)


#Missing Values
print((Data_House.isnull().sum()/len(Data_House)).sort_values(ascending=False)) #--> no missing values


# Show statistics for each colums
#print(Data_clean.describe())
# Built histogram, after displaying histogram we can see that attribut are not scaled
Data_House.hist(figsize=(30,20))
#plt.show()
#Outliers detection
#We create this box plot to see the outliers in price attribits
plt.figure(figsize=(6,4))
sns.boxplot(y=Data_House['price']).set_title
#plt.show() # we observe that this dataset contains the outliers, point that are far of maximum prices

#To compare boxplot attributs we should scale data
#To draw boxplpot without price
dataSub = Data_House[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','sqft_above','sqft_basement','lat','long']]
#dataSub=Data_clean.drop(['price'],axis=1)
scaler= StandardScaler()
dataSub['bedrooms']=scaler.fit_transform(Data_House[['bedrooms']].values)
dataSub['bathrooms']=scaler.fit_transform(Data_House[['bathrooms']].values)
dataSub['sqft_living']=scaler.fit_transform(Data_House[['sqft_living']].values)
dataSub['sqft_lot']=scaler.fit_transform(Data_House[['sqft_lot']].values)
dataSub['floors']=scaler.fit_transform(Data_House[['floors']].values)
dataSub['waterfront']=scaler.fit_transform(Data_House[['waterfront']].values)
dataSub['view']=scaler.fit_transform(Data_House[['view']].values)
dataSub['sqft_above']=scaler.fit_transform(Data_House[['sqft_above']].values)
dataSub['sqft_basement']=scaler.fit_transform(Data_House[['sqft_basement']].values)
dataSub['lat']=scaler.fit_transform(Data_House[['lat']].values)
dataSub['long']=scaler.fit_transform(Data_House[['long']].values)
sns.boxplot(data=dataSub)
#plt.show()

#*******************************************Write a paragraph selecting the most important features (feature selection) **********
# To know the features that are highly correlated
def plot_correlation_map( df ):

    corr = df.corr() # function that returns correlation wth all numeric columns {dataframe}

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 12 , as_cmap = True )

    s = sns.heatmap( # create a heat map with correlation values so we should pass correlation dataframe

        corr,

        cmap = cmap,

        square=True,

        cbar_kws={ 'shrink' : .10 },

        ax=ax,

        annot = True,

        annot_kws = { 'fontsize' : 12 }

        )

    plt.show()
plot_correlation_map(Data_House)
#** with heatmap and pearson correlation we can notice features that are highly correlated with Price_target, we fixed a threshold =0.6 all value superior or equal 0.6 means
# that the value is highly correlated, and we selected as features bedrooms,bathrooms,sqft_living,floors,sqft_above,sqft_basement,saft_lot


array = Data_House.values
# *************************************** j'ai essayé une méthode PCA mais je n'ai vraiment pas pu determiner les fetaures important**********
X = array[:,3:16]
Y_1 = array[:,2]
# feature extraction
pca = PCA(n_components=10)
fit = pca.fit(X)
#Summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

#********************************************drop useless data**********************************************************************
Data_clean= Data_House.drop(['id','date','condition','grade','yr_built','yr_renovated','zipcode','sqft_living15','sqft_lot15'],axis=1)

#*******************************************Splitting data*************************************************************************

train_data,test_data = train_test_split(Data_clean,test_size=0.20,random_state=0)

#********************************** Apply Linear regression***********************************************************************
model=LinearRegression()   #build linear regression model
X_train=np.array(train_data['sqft_living']).reshape(-1,1)
Y_train=np.array(train_data['price']).reshape(-1,1)
model.fit(X_train,Y_train)  #fitting the training data
print('R squared training',round(metrics.r2_score(X_train,Y_train),3))

X_test=np.array(test_data['sqft_living']).reshape(-1,1)
Y_test=np.array(test_data['price']).reshape(-1,1)
predicted=model.predict(X_test) #testing our model’s performance

print("MSE", mean_squared_error(Y_test,predicted))
print("R squared", metrics.r2_score(Y_test,predicted)) #R squared = 0.46 on a une loss function avec un taux d'erreur elevé donc le model avec l'unique features sqft_living n'arrive pas
# a prédire corréctement le prix

#*****************************************Plot Linear regression***************************************************************
_, ax = plt.subplots(figsize= (12, 10))
plt.scatter(X_test, Y_test, color= 'darkgreen', label = 'data')
plt.plot(X_test, model.predict(X_test), color='red', label= ' Predicted Regression line')
plt.xlabel('Living Space (sqft)')
plt.ylabel('price')
plt.legend()
plt.show()

#********************************************Multiple Regression*************************************************************
Multiple_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','sqft_above','sqft_basement','lat','long']
model=LinearRegression()
model.fit(train_data[Multiple_features],train_data['price'])
pred = model.predict(test_data[Multiple_features])

mean_squared_error = metrics.mean_squared_error(Y_test, pred)
print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))
print('R-squared (training) ', round(model.score(train_data[Multiple_features], train_data['price']), 3))
print('R-squared (testing) ', round(model.score(test_data[Multiple_features], test_data['price']), 3))

#Result= R-squared (training)  0.646 R-squared (testing)  0.626 : avec Multiple features on voit que loss function est plus proche que 1 contrairement
# a linear regression est donc predit mieux le targer price
#Conclusion : pour faire un bon apprentissage il faut que le model puisse avoir le maximum de features pour predire un target,avec un seul features le model ne sera puissant
#et n'arrivera a predire le target --> underfitting

#*****************************Apply Polynomial regression and compare it to the linear and multilinear regression. ***************

polyfeat=PolynomialFeatures(degree=2)
xtrain_poly=polyfeat.fit_transform(train_data[Multiple_features])
xtest_poly=polyfeat.fit_transform(test_data[Multiple_features])

poly=LinearRegression()
poly.fit(xtrain_poly,train_data['price'])
polypred=poly.predict(xtest_poly)


mean_squared_error = metrics.mean_squared_error(test_data['price'], polypred)
print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))
print('R-squared (training) ', round(poly.score(xtrain_poly, train_data['price']), 3))
print('R-squared (testing) ', round(poly.score(xtest_poly, test_data['price']), 3))

#R-squared (testing)  0.707 on a la meilleur prédiction
# La regression polynomial nous donne un score R-carré (test) de 0,707. D'après les rapports ci-dessus, nous pouvons conclure que la régression polynomiale pour le degré = 2 est la meilleure solution.