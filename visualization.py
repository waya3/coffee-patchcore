import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, gaussian_kde
from params import RESULTDIR

def kde_minval(name, good_minval, bad_minval=None, FB=None):
    good_minval = np.array(good_minval)
    bad_minval = np.array(bad_minval)
    good_flat = good_minval.flatten()
    bad_flat = bad_minval.flatten()

    # if bad_minval == None:
    #     # x軸生成
    #     x = np.linspace(5,51,300)
    #     kde_good = gaussian_kde(good_flat)
    #     plt.clf()
    #     plt.ylim(0, 0.14)
    #     plt.plot(x, kde_good(x), color='red')
    #     plt.fill_between(x, kde_good(x), alpha=0.2, color='red')
    #     plt.xlabel('Anomaly Score')
    #     plt.ylabel('Density')
    #     plt.title('KDE of Anomaly Scores (Flattened)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f"{RESULTDIR}/kde/{FB}/{name}.jpg")
    #     return
    
    kde_good = gaussian_kde(good_flat)
    kde_bad = gaussian_kde(bad_flat)

    # x軸生成
    x = np.linspace(min(good_flat.min(), bad_flat.min()) - 1,
                    max(good_flat.max(), bad_flat.max()) + 1,
                    300)

    # プロット
    if name == "TwoSides":
        plt.clf()
        plt.ylim(0, 0.14)
        plt.plot(x, kde_good(x), label='Front', color='red')
        plt.plot(x, kde_bad(x), label='Back', color='blue')
        plt.fill_between(x, kde_good(x), alpha=0.2, color='red')
        plt.fill_between(x, kde_bad(x), alpha=0.2, color='blue')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('KDE of Anomaly Scores (Flattened)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTDIR}/kde_result_{name}.jpg")
    else:
        plt.clf()
        plt.ylim(0, 0.14)
        plt.plot(x, kde_good(x), label='Normal', color='blue')
        plt.plot(x, kde_bad(x), label='Anomaly', color='red')
        plt.fill_between(x, kde_good(x), alpha=0.2, color='blue')
        plt.fill_between(x, kde_bad(x), alpha=0.2, color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('KDE of Anomaly Scores (Flattened)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTDIR}/kde_result_{name}.jpg")