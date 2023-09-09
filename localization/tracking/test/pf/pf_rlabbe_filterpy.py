'''
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
'''

from filterpy.monte_carlo import systematic_resample
from numpy.random import uniform
import numpy as np
import scipy.stats
from numpy.random import randn

class ParticleFilter(object):
  ''' Use velocity model as in typical Kalman filter '''

  def __init__(self, 
              N, # number of particles
              dt=1, # timestep
              dim_x=2, # state vector size
              dim_z=2, # # measurement vectore  size
              dim_u=2,
              width=None, height=None, # uniform_particle
              initial_x=None, inital_err=None # gaussian_particle
              ):
    
    self.N = N
    self.dt = dt
    self.dim_x = dim_x
    self.dim_z = dim_z
    self.dim_u = dim_u
    
    self.R = 1. # measurement noise
    self.Q = 1. # process noise

    # create particles and weights
    if initial_x is not None:
        if inital_err is None: inital_err = 1.
        particles = self.create_gaussian_particles(
            mean=initial_x, std=inital_err, N=N)
    else:
        xlim = width
        if isinstance(xlim, int) or isinstance(xlim, float):
            xlim = (0, width)

        ylim = height
        if isinstance(ylim, int) or isinstance(ylim, float):
            ylim = (0, height)
        
        particles = self.create_uniform_particles(xlim, ylim, N)
    weights = np.ones(N) / N

    self.particles = particles
    self.weights = weights
  
  def predict(self, u=None):
    ''' For particle filter, Control inputs (u) are typically considered as velocity:
    x = x + u * dt
    '''
    N = self.N
    dt = self.dt
    particles = self.particles
    Q = self.Q 

    if u is None:
      u = np.zeros((self.dim_u))

    particles += u.reshape((1,-1))*self.dt + np.random.randn(N,self.dim_x)*Q

  def update(self, z):
    ''' measurements are locations '''

    N = self.N
    particles = self.particles
    weights = self.weights
    R = self.R

    z = z.reshape((1,self.dim_z))

    distance = np.linalg.norm(particles - z, axis=1)

    weights *= scipy.stats.norm.pdf(x=distance, loc=0, scale=R)
    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

    # resample if too few effective particles
    if self.neff(weights) < N/2:
        indexes = systematic_resample(weights)
        self.resample_from_index(indexes)
        assert np.allclose(weights, 1/N)

  def estimate(self):
      """returns mean and variance of the weighted particles"""
      particles = self.particles
      weights = self.weights

      mean = np.average(particles, weights=weights, axis=0)
      var  = np.average((particles - mean)**2, weights=weights, axis=0)
      return mean, var

  def neff(self, weights):
      return 1. / np.sum(np.square(weights))

  def resample_from_index(self, indexes):
      particles = self.particles
      weights = self.weights
      particles[:] = particles[indexes]
      weights.resize(len(particles))
      weights.fill (1.0 / len(weights))

  def create_uniform_particles(self, x_range, y_range, N):
      particles = np.empty((N, self.dim_x))
      particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
      particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
      return particles

  def create_gaussian_particles(self, mean, std, N):
      N = self.N
      particles = np.empty((N, self.dim_x))
      particles[:, 0] = mean[0] + (randn(N) * std)
      particles[:, 1] = mean[1] + (randn(N) * std)
      return particles
