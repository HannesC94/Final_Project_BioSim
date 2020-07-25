import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def add_rec_patch(ax, idx_rec, color='r'):

    idx_rec_np = np.array(idx_rec)
    ll = (idx_rec_np[0, 0], idx_rec_np[1, 0]
    width=idx_rec_np[0, 0] - idx_rec_np[0, 1]
    height=idx_rec_np[1, 0] - idx_rec_np[1, 1]
    rect=patches.Rectangle((50, 100), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
