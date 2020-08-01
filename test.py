import state_dicts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *

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

# %% make three state definitions based on assignement to rectangles.
################################################################################
# the first dictionary is defined in state_dicts in "state_dicts.py"
rect_dict_1 = state_dicts.rect_dict_1

# call change_core to shrink the rectangles
ds_2 = 5/360*2*np.pi  # 5degrees
rect_dict_2_core = state_dicts.change_core(rect_dict_1, ds=ds_2)

# call change_core to shrink the rectangles
ds_3 = 45/360*2*np.pi  # 30degrees
rect_dict_3_core = state_dicts.change_core(rect_dict_1, ds=ds_3)

# create state trajectories
s_1 = make_state_traj(phi, psi, rect_dict_1, core_def=False)
s_2 = make_state_traj(phi, psi, rect_dict_2_core, core_def=True)
s_3 = make_state_traj(phi, psi, rect_dict_3_core, core_def=True)


# %% testcell MSM evaluation
#################################################################################
# n_tau_list = [20, 100, 800]
#
# data_1 = calc_MSM_validation_data(n_tau_list, s_1, K=3)
# ref_data_1 = data_1[0]
# tau_data_1 = data_1[1]
# np.save('MSM_validation_s_1.npy', data_1)
# data_2 = calc_MSM_validation_data(n_tau_list, s_2, K=3)
# ref_data_2 = data_2[0]
# tau_data_2 = data_2[1]
# np.save('MSM_validation_s_2.npy', data_2)
#
# data_3 = np.load('MSM_validation_s_3.npy', allow_pickle=True)
# data_3 = calc_MSM_validation_data(n_tau_list, s_3, K=3)
# ref_data_3 = data_3[0]
# tau_data_3 = data_3[1]
# np.save('MSM_validation_s_3.npy', data_3)
#
# data_matrix = np.array([data_1, data_2, data_3])

# %%
################################################################################
n_tau_list = [20, 100, 800]

data_1 = np.load('MSM_validation_s_1.npy', allow_pickle=True)
ref_data_1 = data_1[0]
tau_data_1 = data_1[1]

data_2 = np.load('MSM_validation_s_1.npy', allow_pickle=True)
ref_data_2 = data_2[0]
tau_data_2 = data_2[1]

# data_3 = np.load('MSM_validation_s_3.npy', allow_pickle=True)
data_3 = np.load('MSM_validation_s_1.npy', allow_pickle=True)
ref_data_3 = data_3[0]
tau_data_3 = data_3[1]

data_matrix = np.array([data_1, data_2, data_3])

# %% plot the MSM validation of all state definitions for all three states
################################################################################
title_string = 'Validation of MSM model. Every column shows the results of a different state defition. '

grid_dict = {
    'wspace': 0.0,
    'hspace': 0.0
}
# Parameters passed on to the subplots call
params = {'xscale': 'log',
          'xlim': (5, 400),
          }

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11.69, 12),
                         sharey='row', sharex='col', gridspec_kw=grid_dict, subplot_kw=params)
# add empty subplots to generate one x/y-label for all subplots
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel("time in ps", fontsize=16)
plt.ylabel("occupation probability", fontsize=16)

# iterate over the different state definitions (columns)
for def_idx, data in enumerate(data_matrix):
    ref_data = data[0]
    tau_data = data[1]
    # iterate over the different states (rows)
    for state_idx, ax in enumerate(axes[:, def_idx]):
        state = state_idx+1
        add_MSM_val(tau_data, ref_data, ax, state)
# create legend above all subplots
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, title=title_string, fontsize='x-large')
plt.show()
