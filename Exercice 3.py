
nombre = input("Veuillez introduire un nombre pour calculer :\n ")
mult=1
dictionnaire = {"0": 0}
for i in range(1,int(nombre)+1):
     mult= i*i
     dictionnaire .update({i:mult})

print(dictionnaire)