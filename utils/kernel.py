import numpy as np
from utils.integral import int_trapz

class kernel:
    def __init__(self, h, Dim=None):
        self.h = h
        self.Dim = Dim

    def H(self, pdfs):
        n = pdfs.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sqrt_prod = np.sqrt(pdfs[i] * pdfs[j])
                K_val = int_trapz(sqrt_prod, self.h, Dim=self.Dim)
                K[i, j] = K_val
                K[j, i] = K_val
        return K

    def L2(self, pdfs, gamma=1):
        n = pdfs.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                diff_sq = (pdfs[i] - pdfs[j]) ** 2
                dist = int_trapz(diff_sq, self.h, Dim=self.Dim)
                K_val = np.exp(-gamma * dist)
                K[i, j] = K_val
                K[j, i] = K_val
        return K
        
    def L1(self, pdfs, gamma=1.0):
        n = pdfs.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                abs_diff = np.abs(pdfs[i] - pdfs[j])
                dist = int_trapz(abs_diff, self.h, Dim=self.Dim)
                K_val = np.exp(-gamma * dist)
                K[i, j] = K_val
                K[j, i] = K_val
        return K
    
    def compute(self, pdfs, kind='H', gamma=1.0, c=1.0, degree=2):
        if kind == 'H':
            return self.H(pdfs)
        elif kind == 'L2':
            return self.L2(pdfs, gamma)
        elif kind == 'L1':
            return self.L1(pdfs)
        else:
            raise ValueError(f"Unknown kernel kind: {kind}")
