from pyentropy import SortedDiscreteSystem
import numpy as np
import random

#Formant channels:

X_1 = [316, 637, 395, 218, 450, 194, 292, 400, 371, 122, 353, 56]
X_1 = []
for i in range(100):
    X_1.append(random.randint(0, 1))
x_m_1 = max(X_1) + 1
X_dims_1 = (1, x_m_1)
Ym = 2
Ny_1 = np.array([25, 75])
"""
#EI-neurons:

X_2 = [195, 320, 256, 149, 264, 148, 166, 198, 204, 73, 269, 55]
x_m_2 = max(X_2) + 1
X_dims_2 = (1, 321)

Ny = np.array([3, 9])
"""
# MI calculations
s1 = SortedDiscreteSystem(X_1, X_dims_1, Ym, Ny_1)
s1.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
inf1 = s1.I()
s1.calculate_entropies(method='pt', calc=['HX', 'HXY'])
inf2 = s1.I()
s1.calculate_entropies(method='qe', calc=['HX', 'HXY'])
inf3 = s1.I()
inf_formant = np.array([inf1, inf2, inf3])

"""s2 = SortedDiscreteSystem(X_2, X_dims_2, Ym, Ny)
s2.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
inf1_2 = s2.I()
s2.calculate_entropies(method='pt', calc=['HX', 'HXY'])
inf2_2 = s2.I()
s2.calculate_entropies(method='qe', calc=['HX', 'HXY'])
inf3_2 = s2.I()
inf_tde = np.array([inf1_2, inf2_2, inf3_2])"""

mi_formant = np.sum(inf_formant) / 3.
#mi_tde = np.sum(inf_tde) / 3.
std_formant = np.std(inf_formant)
#std_tde = np.std(inf_tde)

print("MI results:", inf_formant)
print("Average:", mi_formant)
#print("MI results:", inf_tde)
#print("Average:", mi_tde)

