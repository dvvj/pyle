
print(__doc__)

from scipy.special import logit, expit

x = [0.1*x for x in range(1, 10)]
for xx in x:
    print('%.2f -> %.3f' % (xx, logit(xx)))


from typing import List
from numpy import ndarray
import numpy as np

def mse(v:ndarray)->ndarray:
    tmp = expit(v)-0
    return tmp * tmp


mten2ten = np.arange(-10, 10, 0.1)
r = mse(mten2ten)

import matplotlib.pyplot as plt
plt.plot(mten2ten, r)
plt.show()
