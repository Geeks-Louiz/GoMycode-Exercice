import  math as mt
def Exercice_7(value):
    result= mt.sqrt(((2*50*value)/30))
    print(round(result))

valeur = input("Veuillez introduire un nombre pour le calcul :\n ")
Exercice_7(int(valeur))