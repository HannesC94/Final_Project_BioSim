import state_dicts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *

# %% read in data
#path_phi = 'simulation_data/ad_angle_PHI.xvg'
#path_psi = 'simulation_data/ad_angle_PSI.xvg'
# read in the files for psi and phi
#psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
#phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')
#
# %% change representation of angles to radians and get values of df
#psi = psi_raw.values
#psi[:, 1] *= 2*np.pi/360
#phi = phi_raw.values
#phi[:, 1] *= 2*np.pi/360
#
#fig, ax1, ax2 = ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])
#s1_rec_ul = [[-np.pi, -2.07], [1.3, np.pi]]
#s1_rec_ur = [[1.3, np.pi], [1.3, np.pi]]
#s1_rec_ll = [[-np.pi, -2.07], [-np.pi, -2]]
#s1_rec_lr = [[1.3, np.pi], [-np.pi, -2]]
#rec_list_s1 = [s1_rec_ul, s1_rec_ur, s1_rec_ll, s1_rec_lr]
#idx_state1 = where_in_rec(phi, psi, rec_list_s1)
#
#s2_rec = [[-2.07, 1.3], [-np.pi, -2]]
#s2_rec_2 = [[-2.07, 1.3], [1.3, np.pi]]
#rec_list_s2 = [s2_rec, s2_rec_2]
#idx_state2 = where_in_rec(phi, psi, rec_list_s2)
#
#s3_rec = [[-np.pi, np.pi], [-2, 1.3]]
#rec_list_s3 = [s3_rec]
# for rec in rec_list_s1:
#    add_rec_patch(ax1, rec, color='g')
# for rec in rec_list_s2:
#    add_rec_patch(ax1, rec, color='c')
# for rec in rec_list_s3:
#    add_rec_patch(ax1, rec, color='m')
# plt.show()


# %% read in data and prepare for further analysis
################################################################################
path_phi = 'simulation_data/ad_angle_PHI.xvg'
path_psi = 'simulation_data/ad_angle_PSI.xvg'
# read in the files for psi and phi
psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')
# change representation of angles to radians and get values of df
psi = psi_raw.values
psi[:, 1] *= 2*np.pi/360
phi = phi_raw.values
phi[:, 1] *= 2*np.pi/360
# %% make three state definitions based on assignement to rectangles. these are
################################################################################
# defines in "state_dicts.py"
# get dictionaries with the rectangles for the 3 different state definitions
# from the python file state_dicts
rect_dict_1 = state_dicts.rect_dict_1
rect_dict_2 = state_dicts.rect_dict_2
rect_dict_3 = state_dicts.rect_dict_3
rect_dict_3['state1']
rect_dict_2['state1']
# create state trajectories
s_1 = make_state_traj(phi, psi, rect_dict_1, core_def=False)
s_2 = make_state_traj(phi, psi, rect_dict_2, core_def=True)
s_3 = make_state_traj(phi, psi, rect_dict_3, core_def=True)

# %% testcell MSM evaluation
################################################################################
n_tau_list = [5, 50, 200]
K = 3
data_s_1 = np.load('MSM_validation_s_1.npy', allow_pickle=True)
ref_data, tau_data = data_s_1[0], data_s_1[1]

params = {'xscale': 'log',
          'xlim': (5, 5e3),
          'xlabel': 't [ps]/dt'
          }
fig, axes = MSM_validation_plot(tau_data, ref_data, states=[
                                1, 2, 3], params=params, figsize=(10, 10))
plt.tight_layout()
plt.show()
#a = np.load('MSM_validation_s_2.npy', allow_pickle=True)
