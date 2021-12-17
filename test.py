import numpy as np
a2 = np.array([0, 0, 1, 1, 1])
print(a2)
mask = a2<0.5
print(mask)
b2 = a2[mask]
print(b2)
b2[0] =17
print(a2)
