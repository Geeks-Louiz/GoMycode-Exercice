#La fonction qui effectue le factoriel
def Factorielle(n):
    fact=1
    for i in range(1,n+1):
        fact= fact*i
    print("\n", fact)

# On fait appelle á la foction pour calculer le factorielle
while(1): # effectuer le calcul autant de fois qu'on voudrq
 value = input("Veuillez introduire un nombre pour calculer son factoriel :\n ")
 if (int(value) ==0):
    print("\n 0")
 else :
     Factorielle(int(value))