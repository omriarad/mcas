# SYSTEM IMPORTS
from typing import List
import matplotlib.pyplot as plt
import numpy as np


# PYTHON PROJECT IMPORTS

EPOCHS: np.ndarray = np.array([1, 25, 50, 75])
NAMES: List[str] = ["fs-dax", "dev-dax", "dram+memory mode", "memory map"]
make_posterior_times: np.ndarray = np.array(
    [[4.466201305,      110.4730248,    229.9773089,    336.8223925],
     [2.854170958,      70.16877405,    140.2939212,    210.2906936],
     [0.00208346049,    0.00258509318,  0.00187754631,  0.002103249232],
     [0.001129150391,   0.001351833344, 0.001236597697, 0.001342455546]])


total_script_means: np.ndarray = np.array(
    [[157.5966412, 4174.234706, 8684.12876,  12701.56397],
     [155.0929572, 4127.366307, 8333.924554, 12585.12044],
     [48.28884617, 1287.494542, 2741.781028, 4546.528845],
     [75.76046046, 4985.449054, 17129.18271, 37927.03085]])

total_script_stds: np.ndarray = np.array(
    [[2.260823483,  5.454230672, 89.2715892,  80.98970695],
     [0.7610416461, 21.74443245, 15.14409582, 17.91654522],
     [1.486293315,  35.87487558, 60.77230366, 253.0818632],
     [7.483302835,  179.9984051, 224.4659209, 44.53548966]])


avg_epoch_means: np.ndarray = np.array(
    [[147.6738516, 162.4155522, 168.9955877, 164.7875351],
     [149.3040636, 162.1689263, 163.8128882, 164.9562864],
     [41.74385905, 48.10868245, 50.83244295, 56.22091977],
     [68.96501764, 198.2526263, 342.3079129, 505.5171581]])


avg_epoch_stds: np.ndarray = np.array(
    [[0.9122212463, 0.2649968268, 0.8489132814, 0.8025844537],
     [0.7453416315, 0.754876835,  0.3150811819, 0.3170211586],
     [0.6807780085, 3.02863544,   13.43297525,  38.34959738],
     [7.563460717,  12.14549112,  38.37610946,  37.2030336]])


avg_epoch_relative_factors: np.ndarray = np.array(
    [[3.537618584,  3.376013308,    3.324561597,    2.93107149],
     [3.576671322,  3.370886877,    3.222605067,    2.934073065],
     [1,            1,              1,              1],
     [1.652099715,  4.12093236,     6.734044107,    8.991620204]])

total_script_time_relative_factors: np.ndarray = np.array(
    [[3.263624081, 3.242137788, 3.167331261, 2.793683798],
     [3.211775999, 3.205734994, 3.039602532, 2.768072275],
     [1,           1,           1,           1],
     [1.568901857, 3.872209856, 6.24746562,  8.341975196]])


def main() -> None:
    """
    for i, name in enumerate(NAMES):
        plt.errorbar(EPOCHS, total_script_means[i], yerr=total_script_stds[i], label=name,
                     marker="s")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("total runtime (s)")
    plt.show()

    for i, name in enumerate(NAMES):
        plt.errorbar(EPOCHS, avg_epoch_means[i], yerr=avg_epoch_stds[i], label=name,
                     marker="s")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("avg epoch runtime (s)")
    plt.show()
    """

    plt.rcParams.update({'font.size': 16})
    plt.subplot(2,1,1)
    for i, name in enumerate(NAMES):
        plt.plot(EPOCHS, avg_epoch_relative_factors[i], label=name, marker="s")
    plt.ylabel("avg epoch runtime ratio")
    plt.subplot(2,1,2)
    for i, name in enumerate(NAMES):
        plt.plot(EPOCHS, total_script_time_relative_factors[i], label=name, marker="s")
    plt.ylabel("total runtime ratio")
    plt.xlabel("number of epochs")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()

