'''
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
'''

from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from numpy.random import uniform
import numpy as np

class ParticleFilter(object):

  def __init__(self, 
                N, # number of particles
                dt=1., # time interval
                measure_std_err=.1, # R: measurement noise
                process_std_err=(.2,.05), # Q: process noise
                control_input=(0.00, 1.414), # U: control input
                xlim=(0,20), ylim=(0,20), hlim=(0,6.28), # uniform_particle
                initial_x=None, initial_std=(5, 5, np.pi/4) # gaussian_particle
                ):
    self.N = N
    self.dt = dt
    self.measure_std_err = measure_std_err
    self.process_std_err = process_std_err
    self.control_input = control_input

    # create particles and weights
    if initial_x is not None:
        particles = self.create_gaussian_particles(
            mean=initial_x, std=initial_std, N=N)
    else:
        particles = self.create_uniform_particles(xlim, ylim, hlim, N)
    weights = np.ones(N) / N

    self.particles = particles
    self.weights = weights
    
  def predict(self):
    """ move according to control input (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""
    N = self.N
    dt = self.dt
    particles = self.particles
    process_std_err = self.process_std_err 
    control_input = self.control_input 

    # update heading
    particles[:, 2] += control_input[0] + (randn(N) * process_std_err[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (control_input[1] * dt) + (randn(N) * process_std_err[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    
  def update(self, z, landmarks):
    N = self.N
    particles = self.particles
    weights = self.weights
    measure_std_err = self.measure_std_err

    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, measure_std_err).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

    # resample if too few effective particles
    if self.neff(weights) < N/2:
        indexes = systematic_resample(weights)
        self.resample_from_index(particles, weights, indexes)
        assert np.allclose(weights, 1/N)

  def neff(self, weights):
      return 1. / np.sum(np.square(weights))

  def resample_from_index(self, particles, weights, indexes):
      particles[:] = particles[indexes]
      weights.resize(len(particles))
      weights.fill (1.0 / len(weights))

  def estimate(self):
      """returns mean and variance of the weighted particles"""
      particles = self.particles
      weights = self.weights

      pos = particles[:, 0:2]
      mean = np.average(pos, weights=weights, axis=0)
      var  = np.average((pos - mean)**2, weights=weights, axis=0)
      return mean, var

  def create_uniform_particles(self, 
                    x_range, y_range, hdg_range, N):
      particles = np.empty((N, 3))
      particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
      particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
      particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
      particles[:, 2] %= 2 * np.pi
      return particles

  def create_gaussian_particles(self, mean, std, N):
      N = self.N
      particles = np.empty((N, 3))
      particles[:, 0] = mean[0] + (randn(N) * std[0])
      particles[:, 1] = mean[1] + (randn(N) * std[1])
      particles[:, 2] = mean[2] + (randn(N) * std[2])
      particles[:, 2] %= 2 * np.pi
      return particles
