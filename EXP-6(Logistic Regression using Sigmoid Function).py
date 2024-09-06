import numpy as np
import matplotlib.pyplot as plt
  
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  
z = np.arange(-5, 5, 0.1)
plt.plot(z, sigmoid(z))
plt.title('Sigmoid Function')
plt.show()
