import numpy as np
import scipy

class TSAEDistance:
    def __init__(self, arr1, arr2, dt):
        self.delta_t = dt
        self.np1, self.np2 = np.array(arr1), np.array(arr2)
        self.err = np.array([scipy.absolute(scipy.roll(self.np2, dt*i)-self.np1).sum() for i in range(self.np1.size/dt)])
    
    def minDistance(self):
            return self.err.min()
    
    def shiftedNPArray(self):
        return scipy.roll(self.np2, self.delta_t*self.err.argmin())
