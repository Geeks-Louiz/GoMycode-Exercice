
import numpy as np
def convert_array_to_list (np1):
    x=np.array(np1)
    print("Original array elements:")
    print(x)
    print("Array to list:")
    print(x.tolist())

a = np.array([[0,1],[2,3],[4,5]])
convert_array_to_list(a)