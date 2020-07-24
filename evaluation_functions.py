import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def add_rec_patch(ax, idx_rec, color='r'):

    idx_rec_np = np.array(idx_rec)
    ll = (idx_rec_np[0, 0], idx_rec_np[1, 0])
    width = idx_rec_np[0, 1] - idx_rec_np[0, 0]
    height = idx_rec_np[1, 1] - idx_rec_np[1, 0]
    rect = patches.Rectangle(ll, width, height, linewidth=1,
                             edgecolor=color, facecolor='none')
    ax.add_patch(rect)


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
            [[[phi_min1, phi_max1],[psi_min1, psi_max1]],
             [[phi_min2, phi_max2],[psi_min2, psi_max2]] ]

        Returns
        -------
        idx_state: ndarray, shape(N,)
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
    idx_state = np.logical_or.reduce(np.array(both_in_rec_list))
    return idx_state


def ramachandran_plot(phi, psi, xy_idx=[0.5, 0.5], figsize=[12, 5], bins=100):
    '''
        Plot a histogram of Phi and Psi along with a horizontal and a vertical
        line in the histogram. The occupation propability along these lines is
        plotted in a second plot.

        Parameters
        ----------
        phi: ndarray, shape(N,2)
            array with time in first column and phi angles in second column
        psi: ndarray, shape(N,2)
            array with time in first column and psi angles in second column
        xy_idx: list or tuple with values from [0, 1]
            contains the position along x (phi) and y(psi) axis, where the
            vertical and horizontal lines are shown. Value is fraction of
            the respective axis length.
        figsize: tuple/list
            figure size, passed to plt.subplots()
        bins: int
            number of bins in histogram

    '''
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Ramachandran plot and occupation probability\n along two chosen lines.')
    # create histogram data for phi/psi plot
    histogram = ax1.hist2d(phi[:, 1], psi[:, 1], bins=bins, density=True)
    h = histogram[0]
    phi_edges = histogram[1]
    psi_edges = histogram[2]
    phi_idx = int(bins*xy_idx[0])
    psi_idx = int(bins*xy_idx[1])
    phi_c = phi_edges[phi_idx]
    psi_c = psi_edges[psi_idx]

    ax1.hlines(y=psi_c, xmin=-np.pi, xmax=np.pi, color='b')
    ax1.vlines(x=phi_c, ymin=-np.pi, ymax=np.pi, color='r')
    ax1.set_xlabel('Phi [rad]')
    ax1.set_ylabel('Psi [rad]')
    #add_rec_patch(ax1, [[-np.pi, np.pi], [-2, 1.3]])
    ax2.plot(phi_edges[:-1], h[:, psi_idx], label='Psi:{:.3f}'.format(psi_c))
    ax2.plot(psi_edges[:-1], h[phi_idx, :], label='Phi:{:.3f}'.format(phi_c), color='r')
    ax2.set_ylim(0, np.max(h))
    ax2.set_xlabel('Phi/Psi [rad]')
    ax2.set_ylabel('occupation propability along line')
    fig.colorbar(histogram[3], ax=ax1)

    plt.legend()
    # plt.tight_layout()
    # plt.show()
    return fig, ax1, ax2


# def make_plot():
#    fig, ax = plt.subplots()
#    ax.plot(np.arange(10), np.arange(10))
#    return fig, ax
#
#
#f, a = make_plot()
# a.set_xlabel('asd')
# plt.show()
#
#add_rec_patch(ax, [[2, 4], [3, 4]])
#np.array([[2, 4], [3, 4]])[1, 0]
