#Import des bibliotheques
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# *****************************************************Début traitement ************************************************************
#                                                       Importing data
d = pd.read_csv('titanic-passengers.csv',encoding='UTF-8',delimiter= ';')
#               I changed the value of attribut Survived on numeric value
d["Survived"]= d["Survived"].replace('No', 0)
d["Survived"]= d["Survived"].replace('Yes', 1)

#start by showing the head of the dataset then some general information about the data columns and values


print(d.head())

#*************************************************** Pré-traitement**************************************************************#
print(d['Cabin'].head().isnull())
# Le nombre de valeurs manquantes sur chaques colonnes
print(d.isnull().sum())
# Le nombre de valeurs manquantes totales
print(d.isnull().sum().sum())


# Remplacement des valeurs manquantes par la moyenne
d['Age'].fillna(d['Age'].mean(), inplace=True)
#Remplacement des valeurs manquantes de l'attribut Cabin par la valeurs la plus fréquentes: B96,B98 ou G6
#print(d['Cabin'].value_counts())
d['Cabin'].fillna('B96',inplace=True)
#Remplacement des valeurs manquantes de l'attribut Embarked par la valeurs la plus fréquentes: S
#print(d['Embarked'].value_counts())
d['Embarked'].fillna('S',inplace=True)

#Suppression des columns Cabin et Ticket
titanic_d_clean=d.drop(['Cabin','Ticket'], 1)


#Pour afficher les informations de chqaue colonnes
print(d.info())

#Distribution des colonnes numériques uniquement
print(d.describe())



# **************************************************** Data Visoualization *********************************************************

# Visualize the correlation between Sex and Age in a plot of your choosing, the visualized plot should give us obvious deductions concerning the importance of age and Sex in the survival of the individuals.
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = titanic_d_clean[titanic_d_clean['Sex']=='female']
men = titanic_d_clean[titanic_d_clean['Sex']=='male']

ax = sns.distplot(women[women['Survived']== 1].Age.dropna(), bins=18, label = 'survived', ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']== 0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']== 1].Age.dropna(), bins=18, label = 'survived', ax = axes[1], kde =False)
ax = sns.distplot(men[men['Survived']== 0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[1], kde =False)
ax.legend()
ax.set_title('Male')
plt.show()

# Choose another parametrs
grid = sns.FacetGrid (titanic_d_clean, col = 'Survived', row = 'Pclass')
grid.map (plt.hist, 'Age', bins = 20)
grid .add_legend ()
plt.show()

#Using function plot correlation to draw heatmap

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
plot_correlation_map(titanic_d_clean)

# Try running the function, analyse what does it do exactly and what are the utilities of it, write a paragraph describing your analysis (the paragraph should be in english)

#The heatmap graph calculate the correlation between two attribute if the two attribute have same value (1),if the two attribute are closest the value approach 1
# if the two attribute is so differents like (one attribute have string value, and the second have numeric value) the value in heatmap will be negative and approach -1



#Use the groupby function combined with the mean() to view the relation between Pclass and survived

print(titanic_d_clean[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

# Creation of new column Title
titanic_d_clean['Title'] = 0
titanic_d_clean['Title'] = titanic_d_clean['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_d_clean['Title'] = titanic_d_clean['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', \
                                             'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic_d_clean['Title'] = titanic_d_clean['Title'].replace('Mlle', 'Miss')
titanic_d_clean['Title'] = titanic_d_clean['Title'].replace('Ms', 'Miss')
titanic_d_clean['Title'] = titanic_d_clean['Title'].replace('Mme', 'Mrs')
print(titanic_d_clean)

#Drop Name column in final step

titanic_usless_col= titanic_d_clean.drop('Name',1)
cols = titanic_usless_col.columns.tolist()
#print(cols)  ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'] without name

#Finally, use the Parch and the SibSp columns to create a more useful feature, let's call it FamilySize.

titanic_usless_col['Family_Size'] = 0
titanic_usless_col['Family_Size'] = titanic_usless_col['Parch']+titanic_usless_col['SibSp']
sns.factorplot(x ='Family_Size', y ='Survived', data = titanic_usless_col)
plt.show()

#Explication
#Family_Size denotes the number of people in a passenger’s family. It is calculated by summing the SibSp and Parch columns of a respective passenger.
#Yes the nee attribut is useful because we can notice if  the family size is greater than 5, chances of survival decreases considerably.