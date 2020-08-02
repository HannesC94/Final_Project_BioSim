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

# %% 2.2 visualisation of psi/phi plane
################################################################################
plot = ramachandran_plot(phi, psi, xy_idx=[0.3, 0.92])

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

# %% plot s_1
################################################################################
tau_range_1 = np.arange(1, 200, 5)
tau_range_2 = np.arange(1, 900, 10)
plot_timp(tau_range_1, tau_range_2, s_1, axh_y=49, axv_x=19)

# %% show new state def rectangles
################################################################################
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

visualize_states(rect_dict_2_core, phi, psi, ax=ax1, color_list=['r', 'g', 'y'])
visualize_states(rect_dict_3_core, phi, psi, ax=ax2, color_list=['r', 'g', 'y'])

plt.tight_layout()
plt.show()


# %% compare timp after core definition
################################################################################
tau_range = np.arange(1, 200, 5)
compare_timp(tau_range, [s_1, s_2, s_3], ls=':', marker='o')


# %% testcell MSM evaluation
################################################################################
tau_list = [2, 5, 20]
dt = 0.2
n_tau_list = [int(tau/dt) for tau in tau_list]

data_1 = calc_MSM_validation_data(n_tau_list, s_1, 3)

data_2 = calc_MSM_validation_data(n_tau_list, s_2, 3)

data_3 = calc_MSM_validation_data(n_tau_list, s_3, 3)

data_matrix = np.array([data_1, data_2, data_3])

# %%
################################################################################
plot_MSM_val(data_matrix, ref_points=100)
#fig, axes = plt.subplots(3,3)
# for i in range(3):
#    for j in range(3):
#        add_MSM_val(data_1[1], data_1[0], axes[i,j], transition=(i,j))

np.log10(0.2)

\begin{align}
p(t +\tau) = &  (p_1(t), p_2(t), p_3(t))
\left(begin{array}{ccc}
      P_{11}(\tau) & P_{12}(\tau) & P_{13}(\tau)
      P_{21}(\tau) & P_{22}(\tau) & P_{23}(\tau)
      P_{31}(\tau) & P_{32}(\tau) & P_{33}(\tau)
      \end{array}
      \right)\\
    & \left(begin{array}{c}
            p_1(t)P
            2
            3
            \end{array}
            \right)
\end{align}
