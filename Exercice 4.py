
texte = input(" Veuillez introduire une chaine de caractere  :\n ")
index = input(" Veuillez introduire une l'index de la position a supprimer : \n ")
str = " "
if (int(index) > len(texte)):
    print("L'index est plus long que la chaine de caractere,veuillez vérifier")

else:
 texte_list=list(texte.strip())

 for i in texte_list:

    if(int(index) == texte_list.index(i)):
        texte_list.pop(int(index))

print(''.join(texte_list))
