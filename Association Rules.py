import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import association_rules
import squarify


data = pd.read_csv('Market_Basket_Optimisation.csv',encoding='UTF-8',delimiter= ',')
data.head()
data.fillna(0,inplace=True) # on remplace les nan avec la valeur 0
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
plt.rcParams['figure.figsize'] = (10,6)
color = plt.cm.inferno(np.linspace(0,1,20))
data[0].value_counts().head(20).plot.bar(color = color)
plt.title('Top 20 Most Frequent Items')
plt.ylabel('Counts')
plt.xlabel('Items')
plt.show()
plt.rcParams['figure.figsize']=(10,10)
Items = data[0].value_counts().head(20).to_frame()
size = Items[0].values
lab = Items.index
color = plt.cm.copper(np.linspace(0,1,20))
squarify.plot(sizes=size, label=lab, alpha = 0.7, color=color)
plt.title('Tree map of Most Frequent Items')
plt.axis('off')
plt.show()

#Let’s select itemsets with a minimum of 60% Support
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x : len(x))
print(frequent_itemsets.head(20))

print(frequent_itemsets[frequent_itemsets['length'] >= 2].head(20))
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
print(rules.head(50))
pritn(rules[(rules['lift'] >= 5) & (rules['confidence'] >= 0.4)])
