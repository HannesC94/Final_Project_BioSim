import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% read in data
path_phi = 'ad_angle_PHI.xvg'
path_psi = 'ad_angle_PSI.xvg'
# read in the files for psi and phi
psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')

# %% change representation of angles to radians and get values of df
psi = psi_raw.values
psi[:, 1] *= 2*np.pi/360
phi = phi_raw.values
phi[:, 1] *= 2*np.pi/360

# %%
# define two indices, which are going to be used to get a psiedge/phiedge value. along these values the intensity of the rachmandran plot ist shown
phi_idx, psi_idx = [70, 72]
# define figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[12, 5])
# create histogram data for phi/psi plot
histogram = ax1.hist2d(phi[:, 1], psi[:, 1], bins=100, density=True)
h = histogram[0]
phi_edges = histogram[1]
psi_edges = histogram[2]
phi_c = phi_edges[phi_idx]
psi_c = psi_edges[psi_idx]

ax1.hlines(y=psi_c, xmin=-np.pi, xmax=np.pi, color='b')
ax1.vlines(x=phi_c, ymin=-np.pi, ymax=np.pi, color='r')
ax1.set_xlabel('Phi [rad]')
ax1.set_ylabel('Psi [rad]')
ax2.plot(phi_edges[:-1], h[:, psi_idx], label='Psi:'+str(psi_c))
ax2.plot(psi_edges[:-1], h[phi_idx, :], label='Phi:'+str(phi_c), color='r')
ax2.set_ylim(0, np.max(h))
fig.colorbar(histogram[3], ax=ax1)

plt.legend()
plt.tight_layout()
plt.show()


# %% define rectangular region for the different states
# index files idx_rec = [[phi_min, phi_max], [psi_min, psi_max]]
s3_rec = [[-np.pi, np.pi], [-2, 1.3]]

psi_is_3 = np.asarray((psi[:, 1] >= s3_rec[1][0]) & (psi[:, 1] < s3_rec[1][1]))
idx_state3 = np.copy(psi_is_3)

s2_rec = [[-2.07, 1.3], [-np.pi, -2]]
s2_rec_2 = [[-2.07, 1.3], [1.3, np.pi]]

phi_is_2 = np.asarray((phi[:, 1] >= s2_rec[0][0]) & (phi[:, 1] < s2_rec[0][1]))
psi_is_2 = np.asarray((psi[:, 1] >= s2_rec[1][0]) & (psi[:, 1] < s2_rec[1][1]))
# combine with index for other rectangle
phi_is_2 = np.logical_or(phi_is_2, np.asarray(
    (phi[:, 1] >= s2_rec_2[0][0]) & (phi[:, 1] < s2_rec_2[0][1])))
psi_is_2 = np.logical_or(psi_is_2, np.asarray(
    (psi[:, 1] >= s2_rec_2[1][0]) & (psi[:, 1] < s2_rec_2[1][1])))

idx_state2 = np.logical_and(phi_is_2, psi_is_2)
#s2_phi = [-2.07, 1.3]
#s2_psi = [[-np.pi, -2], [1.3, np.pi]]

#phi_is_2 = np.asarray((phi[:, 1] >= s2_phi[0]) & (phi[:, 1] < s2_phi[1]))
#psi_is_2 = np.asarray((psi[:, 1] >= s2_psi[0][0]) & (psi[:, 1] < s2_psi[0][1]))
# psi_is_2_ = np.asarray((psi[:, 1] >= s2_psi[1][0]) & (psi[:, 1] < s2_psi[1][1]))  # , 1, psi_is_2)
#idx_state2 = np.logical_and(phi_is_2, np.logical_or(psi_is_2, psi_is_2_))
# set values for the state trajectory

s1_rec_ul = [[-np.pi, -2.07], [1.3, np.pi]]
s1_rec_ur = [[1.3, np.pi], [1.3, np.pi]]
s1_rec_ll = [[-np.pi, -2.07], [-np.pi, -2]]
s1_rec_lr = [[1.3, np.pi], [-np.pi, -2]]
rec_list = [s1_rec_ul, s1_rec_ur, s1_rec_ll, s1_rec_lr]


def where_in_rec(phi, psi, rec_list):
    '''
        calculates boolean array of indices where [phi, psi] are in a a rectangle.

        Parameters
        ----------
        phi: ndarray, shape(N,2)
            array containing phi values in second column and time in first column
        psi: ndarray, shape(N,2)
            array containing psi values in second column and time in first column
        rec_list: list
            list of lists, which contain the indices, that define the rectangles
            [[[phi_min1, phi_max1],[psi_min1, psi_max1]],[[phi_min2, phi_max2],[psi_min2, psi_max2]] ]

        Returns
        -------
        idx_array: ndarray, shape(N,)
            boolean array. True where [phi[i,1], psi[i,1]] is in any of the
            rectangles, specified in rec_list
    '''
    phi_in_rec_list = []
    psi_in_rec_list = []
    both_in_rec_list = []
    for rec in rec_list:
        phi_in_rec = np.asarray((phi[:, 1] >= rec[0][0]) & (phi[:, 1] < rec[0][1]))
        psi_in_rec = np.asarray((psi[:, 1] >= rec[1][0]) & (psi[:, 1] < rec[1][1]))
        phi_in_rec_list.append(phi_in_rec)
        psi_in_rec_list.append(psi_in_rec)
        both_in_rec_list.append(np.logical_and(phi_in_rec, psi_in_rec))
    idx_state1 = np.logical_or.reduce(np.array(both_in_rec_list))


s = np.zer s(psi.shape)
s[:, 0] = psi[:, 0]

s[idx_state2, 1] = 2
np.all(s[idx_state3, 1] == 0)
s[idx_state3, 1] = 3
idx_state1 = np.asarray(s[:, 1] == 0)
s[idx_state1, 1] = 1
np.all(s[:, 1] != 0)

plt.scatter(phi[idx_state1, 1], psi[idx_state1, 1], label='State 1')
plt.scatter(phi[idx_state2, 1], psi[idx_state2, 1], label='State 2')
plt.scatter(phi[idx_state3, 1], psi[idx_state3, 1], label='State 3')
plt.legend()
plt.xlabel('Phi [rad]')
plt.ylabel('Psi [rad]')
plt.show()
# %% calculate transition matrix


def T_of_tau(s, tau, K):
    N_ij = np.zeros((K, K))
    for i in np.arange(K)+1:
        delta_si = np.array(s[:, 1] == i).astype(int)
        for j in np.arange(K)+1:
            delta_sj = np.array(s[:, 1] == j).astype(int)
            N_ij[i-1, j-1] = delta_si[:-tau].dot(delta_sj[tau:])
    # normalize N_ij
    N_ij = N_ij/np.sum(N_ij, axis=1)[:, np.newaxis]
    return N_ij


def get_lambda2(A):
    """
    Calculates the
    """
    eig_values, eig_vectors = np.linalg.eig(A)
    lambda2 = np.sort(eig_values)[-2]
    return lambda2


def calc_timp(tau_range, s, K):
    timp = np.empty(len(tau_range))
    for i, tau in enumerate(tau_range):
        T_tau = T_of_tau(s, tau, K)
        lambda2 = get_lambda2(T_tau)
        timp[i] = - tau/np.log(lambda2)
    return timp


tau_range = np.arange(1, 300, 2)
timp = calc_timp(tau_range, s, K=3)
timp
plt.plot(tau_range, timp)
psi_raw
