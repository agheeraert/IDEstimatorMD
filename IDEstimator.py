import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import lhsmdu
from scipy.spatial.distance import cdist
import diversipy.hycusampling as hs
from scipy.stats import cauchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle as pkl

class IDEstimator:
    def __init__(self, distances, discard_tail=True):
        """ distances is a NxN matrix representing the distances between all the i and j of the data set
        and output is the output path"""
        if distances.shape[0] == distances.shape[1]:
            self.distances = distances
            self.N = distances.shape[0]
            self.discard_tail = discard_tail
        else:
            print('Distances not an NxN matrix')
    
    def mu(self):
        """returns mu for each point of the dataset"""
        ordered_distances = np.sort(self.distances)
        mu = ordered_distances[:,2] / ordered_distances[:,1]
        return mu
    
    
    def fit(self, output):
        # X = np.log(self.empir_cumulate())
        # Y = -np.log(1-np.arange(self.N)/self.N)
        mu = self.mu()
        sort_idx = np.argsort(mu)
        Femp = np.arange(self.N)/self.N
        lr = LinearRegression(fit_intercept=False)
        if self.discard_tail:
            lr.fit(np.log(mu[sort_idx][:int(0.9*self.N)]).reshape(-1, 1), -np.log(1-Femp[:int(0.9*self.N)]).reshape(-1,1))
            r2 = lr.score(np.log(mu[sort_idx][:int(0.9*self.N)]).reshape(-1, 1), -np.log(1-Femp[:int(0.9*self.N)]).reshape(-1,1))
            # polyfit = np.poly1d(np.polyfit(np.log(mu[sort_idx][:int(0.9*self.N)]), -np.log(1-Femp[:int(0.9*self.N)]), deg=2))
        else:
            lr.fit(np.log(mu[sort_idx]).reshape(-1, 1), -np.log(1-Femp).reshape(-1,1))
            r2 = lr.score(np.log(mu[sort_idx]).reshape(-1, 1), -np.log(1-Femp).reshape(-1,1))
            # polyfit = np.poly1d(np.polyfit(np.log(mu[sort_idx]), -np.log(1-Femp), deg=2))
        plt.figure()
        slope = lr.coef_[0][0]
        plt.scatter(np.log(mu[sort_idx]), -np.log(1-Femp), c='r', label='data', marker='+')
        plt.plot(np.log(mu[sort_idx]), lr.predict(np.log(mu[sort_idx]).reshape(-1,1)), c='k', label='linear fit', linewidth=0.2)
        # plt.plot(np.log(mu[sort_idx]), polyfit(np.log(mu[sort_idx])), c='g', label='polynomial fit', linewidth=0.5)
        plt.title("ID = "+str(round(slope,3))+" ; $r^2=$"+str(round(r2, 4)))
        plt.savefig(output)
        pkl.dump(mu, open(output.replace('.png', '.p'), 'wb'))

        

def hypercube_surface(dimension):
    """Draws a point on a unit hypercube in dimension n"""
    d = np.random.randint(dimension)
    d_value = np.random.randint(2)
    point = np.random.random(size=dimension)
    point[d]=d_value
    return point

def hypercube(dimension, n_sample):
    return np.random.random(size=(n_sample, dimension))

def hypersample(dimension, nsample):
    sample = np.zeros((dimension, nsample))
    for i in range(dimension):
        sample[i] = np.arange(nsample)
    return sample

def swiss_roll(length_phi, length_Z, sigma, nsample):
    phi = length_phi*np.random.rand(nsample)
    xi = np.random.rand(nsample)
    Z = length_Z*np.random.rand(nsample)
    X = 1./6*(phi + sigma*xi)*np.sin(phi)
    Y = 1./6*(phi + sigma*xi)*np.cos(phi)
    swiss_roll = np.array([X, Y, Z]).transpose()
    return swiss_roll


if __name__ == '__main__':
    points = hs.lhd_matrix(20000, 14)/20000
    # points = swiss_roll(15, 15, 0, 20000)
    # points = np.random.standard_cauchy((20000,20))/20000

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:,0], points[:,1], points[:,2], s = 1, c='black')
    # plt.savefig('/home/aghee/IDEstimatorMD/results/hypercube_distrib.png')
    # plt.close()
    # distances = cdist(points, points)
    # IDEstimator(distances).fit('/home/aghee/IDEstimatorMD/results/swiss_roll_test.png')
    # points = hs.lhd_matrix(2500, 14)
    # points = cauchy.pdf(np.random.random((20000,20)))
    # points = np.random.normal(size=(20000,20))
    
    distances = cdist(points, points)
    IDEstimator(distances, discard_tail=True).fit('/home/aghee/IDEstimatorMD/results/hypercube_test.png')
