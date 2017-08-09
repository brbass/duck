import numpy as np
from diffusion_solution import diffusion_solution

class Material:
    def __init__(self,
                 sigma_t,
                 sigma_s0,
                 sigma_s1,
                 source):
        self.sigma_t = sigma_t
        self.sigma_s0 = sigma_s0
        self.sigma_s1 = sigma_s1
        self.source = source
        self.sigma_a = sigma_t - sigma_s0
        self.diff = 1. / (3. * (sigma_t - sigma_s1))
        
class Boundary:
    def __init__(self,
                 psi):
        self.psi = psi
        self.current = 0.5 * psi

class Mesh:
    def __init__(self,
                 lengths,
                 materials,
                 boundaries):
        self.lengths = lengths
        self.total_length = lengths[0] + lengths[1]
        self.center_location = lengths[0]
        self.boundary_locations = [0., self.total_length]
        self.materials = materials
        self.boundaries = boundaries
        
    def index(self,
              x):
        if 0 < x:
            print("position {} outside of problem".format(x))
        elif x <= self.lengths[0]:
            return 0
        elif x > self.total_length:
            print("position {} outside of problem".format(x))
        else:
            return 1
        
    def material(self,
                 x):
        return self.materials[self.index(x))
        
class Solution:
    def __init__(self,
                 mesh):
        self.mesh = mesh
    
    def val(self,
            x):
        xb = self.mesh.boundary_locations
        xa = self.mesh.center_location
        df = [self.mesh.materials[0].diff, self.mesh.materials[1].diff]
        sa = [self.mesh.materials[0].sigma_a, self.mesh.materials[1].sigma_a]
        q = [self.mesh.materials[0].source, self.mesh.materials[1].source]
        j = [self.mesh.boundaries[0].current, self.mesh.boundaries[1].current]
        
        return diffusion_solution(x,
                                  xb,
                                  xa,
                                  df,
                                  sa,
                                  q,
                                  j)

