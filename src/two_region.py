import numpy as np

class Cross_Section:
    def __init__(self,
                 val1,
                 val2):
        self.val1 = val1
        self.val2 = val2
    def val(self, x):
        if x < 1:
            return self.val1
        else:
            return self.val2

class Solution:
    def __init__(self,
                 sigma_t,
                 source,
                 psi0,
                 mu):
        self.sigma_t = sigma_t
        self.source = source
        self.psi0 = psi0
        self.mu = mu
    
    def val(self,
            x):
        mu = self.mu
        
        if x <= 1:
            f0 = self.psi0
            st1 = self.sigma_t.val1
            s1 = self.source.val1
            k = np.exp(-st1 * x / mu)
            return k * f0 + s1 / st1 * (1 - k)
        else:
            f1 = self.val(1)
            st2 = self.sigma_t.val2
            s2 = self.source.val2
            x2 = x - 1
            k = np.exp(-st2 * x2 / mu)
            return f1 * k +s2 / st2 * (1 - k)
    # def dval(self,
    #          x):
    #     st1 = self.sigma_t.val1
    #     st2 = self.sigma_t.val2
    #     s1 = self.source.val1
    #     s2 = self.source.val2
    #     f0 = self.psi0

    #     if x < 1:
    #         return np.exp(-st1 * x) * (s1 - f0 * st1)
    #     else:
    #         return np.exp(-st1 + st2 - st2 * x) * ((s1 - f0 * st1) * st2 + np.exp(st1) * (s2 * st1 - s1 * st2)) / st1

    # def ddval(self,
    #           x):
    #     st1 = self.sigma_t.val1
    #     st2 = self.sigma_t.val2
    #     s1 = self.source.val1
    #     s2 = self.source.val2
    #     f0 = self.psi0

    #     if x < 1:
    #         return np.exp(-st1 * x) * st1 * (-s1 + f0 * st1)
    #     else:
    #         return np.exp(-st1 + st2 - st2 * x) * st2 * (-s1 * st2 + f0 * st1 * st2 + np.exp(st1) * (-s2 * st1 + s1 * st2)) / st1
        
        
