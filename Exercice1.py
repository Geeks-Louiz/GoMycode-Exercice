result = []
# le range doit s'effectuer entre 2000 et 3200 et comme dans la foction range (debut,fin-1) donc on doit incrementer á 3201
for x in range(2000, 3201):
 if (x % 7 == 0) and (x % 5 != 0):
    result.append(x)
print(result)
