'''
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
'''

from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
from pf_rlabbe_filterpy import ParticleFilter

''' Make it purely position based '''

def run_pf1(N, iters=18, sensor_std_err=.1, 
            do_plot=True, plot_particles=False,
            xlim=(0, 20), ylim=(0, 20),
            initial_x=None):
    
    plt.figure()

    pf = ParticleFilter(N, width=xlim, height=ylim, initial_x=initial_x)
    pf.R = sensor_std_err

    if plot_particles:
        particles = pf.particles
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)           
        plt.scatter(particles[:, 0], particles[:, 1], 
                    alpha=alpha, color='g')
    
    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos += (1, 1)

        # move diagonally forward to (x+1, x+1)
        pf.predict()
        
        # incorporate measurements
        pf.update(z=robot_pos)        
            
        mu, var = pf.estimate()
        xs.append(mu)

        if plot_particles:
            particles = pf.particles
            plt.scatter(particles[:, 0], particles[:, 1], 
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    
    xs = np.array(xs)
    #plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()

from numpy.random import seed
seed(2) 
# run_pf1(N=5000, plot_particles=False)

# run_pf1(N=5000, iters=8, plot_particles=True, xlim=(0,8), ylim=(0,8))

# run_pf1(N=100000, iters=8, plot_particles=True,  xlim=(0,8), ylim=(0,8))

seed(6) 
# run_pf1(N=5000, plot_particles=True, ylim=(-20, 20))
run_pf1(N=5000, plot_particles=True, ylim=(-20, 20), initial_x=(1,1))