from IDEstimator import IDEstimator
from os.path import join as jn
import mdtraj as md
import numpy as np
from tqdm import tqdm
import pyemma
import pickle as pkl

class IDEstimatorMD():
    def __init__(self, trajs, topo, methods, output, string="", discard_tail=True, split_chains=None, chains_label=None):
        """
        trajs: list of MD trajectories
        topo: MD topology
        methods: methods employed to compute the distances 
        output: output folder for the results
        string: optional string to better characterize the output files"""
        self.methods = methods
        self.output = output
        self.string = string
        self.trajs = trajs
        self.topo = topo
        self.discard_tail = discard_tail
        self.chains_label = chains_label
        if split_chains:
            self.split_chains = [range(split_chains[0])]
            self.n_chains = len(split_chains)+1
            for i in range(1, len(split_chains)-1):
                self.split_chains.append(range(split_chains[i], split_chains[i+1]))
        else:
            self.split_chains = None

    def plot_ID(self):
        for method in self.methods:
            if method == "TICA":
                distances = self.tica()
            elif method == "RMSD":
                distances = self.rmsd()
            # IDEstimator(distances).fit(jn(self.output, self.string+method+'_fit.png'))
            if self.split_chains:
                for i, chain in enumerate(self.chains_label):
                    IDEstimator(distances[i], discard_tail=self.discard_tail).fit(jn(self.output, self.string+method+'_chain'+chain+'_fit.png'))
            else:
                IDEstimator(distances, discard_tail=self.discard_tail).fit(jn(self.output, self.string+method+'_fit.png'))
    
    def rmsd(self):
        traj = md.load(self.trajs, top=self.topo)
        N = traj.n_frames
        # if self.split_chains:
        #     distances = np.zeros((self.n_chains, N, N))
        # else:
        #     distances = np.zeros((N, N))
        # for frame in tqdm(range(N)):
        #     if self.split_chains:
        #         for chain, elt in enumerate(self.split_chains):
        #             print(chain)
        #             distances[chain, frame] = md.rmsd(traj, traj, frame=frame, atom_indices=elt)
        #             distances[chain, frame, frame] = 0
        #         distances[-1, frame] = md.rmsd(traj, traj, frame=frame, atom_indices=[elt[-1], traj.topology.n_residues])
        #         distances[-1, frame, frame] = 0
        #         print(np.sum(distances==0))
        #     else:
        #         distances[frame]=md.rmsd(traj, traj, frame=frame)
        if self.split_chains:
            distances = np.zeros((self.n_chains, N, N))
            self.split_chains.append(range(self.split_chains[-1][-1], traj.topology.n_residues))
            for chain, atoms in enumerate(self.split_chains):
                for frame in tqdm(range(N)):
                    distances[chain, frame] = md.rmsd(traj, traj, frame=frame, atom_indices=atoms)
                    distances[chain, frame, frame] = 0
        else:
            distances = np.zeros((N, N))
            for frame in tqdm(range(N)):
                distances[frame] = md.rmsd(traj, traj, frame=frame)
        return distances
    
    def tica(self):
        # feat = pyemma.coordinates.featurizer(self.topo)
        # feat.add_distances_ca()
        # traj = pyemma.coordinates.load(self.trajs, features=feat, top=self.topo)
        traj = pyemma.coordinates.load(self.trajs, top=self.topo).transpose()
        distances = pyemma.coordinates.tica(traj)
        return distances.eigenvectors
    
    def save_distances(self, output):
        for method in self.methods:
            if method == "TICA":
                distances = self.tica()
            elif method == "RMSD":
                distances = self.rmsd()
            pkl.dump(distances, open(output, "wb"))
    




if __name__ == '__main__':
    test = IDEstimatorMD('/home/aghee/PDB/prot_apo_sim1_s10.dcd', '/home/aghee/PDB/prot.prmtop', ['RMSD'], '/home/aghee/IDEstimatorMD/results/')
    # test.plot_ID()
    test.save_distances('results/dist_rmsd_apo1.p')




    
