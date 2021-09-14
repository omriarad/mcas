# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os


# PYTHON PROJECT IMPORTS


def load_data(d: str) -> np.ndarray:
    dram_data: np.ndarray = np.loadtxt(os.path.join(d, "dram.txt"))
    pymm_data: np.ndarray = np.loadtxt(os.path.join(d, "pymm.txt"))

    return np.stack([dram_data, pymm_data], axis=0)


def main() -> None:
    cd: str = os.path.abspath(os.path.dirname(__file__))
    mnist_dir: str = os.path.join(cd, "experiments", "mnist", "results")
    cifar_dir: str = os.path.join(cd, "experiments", "cifar10", "results")

    mnist_results: np.ndarray = load_data(mnist_dir)
    cifar_results: np.ndarray = load_data(cifar_dir)

    print("loaded data (mnist.shape, cifar.shape):")
    print(mnist_results.shape, cifar_results.shape)

    # features are stored as [data_loading_time,
    #                         covertree_loading_time,
    #                         dyadic_tree_construction_time,
    #                         wavelet construction time
    #                        ]

    # bar chart hyperparameters
    bar_width=0.35
    opacity=0.8
    xaxis = np.arange(2)

    ###########################################
    # data loading time
    dram_data = np.stack([mnist_results[0,:,0], cifar_results[0,:,0]], axis=0)
    pymm_data = np.stack([mnist_results[1,:,0], cifar_results[1,:,0]], axis=0)
    plt.bar(xaxis,
            # [np.mean(mnist_results[0,:,0]), np.mean(cifar_results[0,:,0])],
            np.mean(dram_data, axis=-1),
            bar_width, alpha=opacity, label="dram")
    plt.bar(xaxis+bar_width,
            # [np.mean(mnist_results[1,:,0]), np.mean(cifar_results[1,:,0])],
            np.mean(pymm_data, axis=-1),
            bar_width, alpha=opacity, label="pymm")
    plt.ylabel("data loading time (s)")
    plt.xticks(xaxis+bar_width, ["mnist", "cifar10"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # wavelet construction time
    dram_data = np.stack([mnist_results[0,:,3], cifar_results[0,:,3]], axis=0)
    pymm_data = np.stack([mnist_results[1,:,3], cifar_results[1,:,3]], axis=0)
    plt.bar(xaxis,
            # [np.mean(mnist_results[0,:,3]), np.mean(cifar_results[0,:,3])],
            np.mean(dram_data, axis=-1),
            bar_width, alpha=opacity, label="dram")
    plt.bar(xaxis+bar_width,
            # [np.mean(mnist_results[1,:,3]), np.mean(cifar_results[1,:,3])],
            np.mean(pymm_data, axis=-1),
            bar_width, alpha=opacity, label="pymm")
    plt.ylabel("wavelet construction time (s)")
    plt.xticks(xaxis+bar_width, ["mnist", "cifar10"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # total pymm/data loading time
    dram_data = np.stack([mnist_results[0,:,3] + mnist_results[0,:,0],
                          cifar_results[0,:,3] + cifar_results[0,:,0]], axis=0)
    pymm_data = np.stack([mnist_results[1,:,3] + mnist_results[1,:,0],
                          cifar_results[1,:,3] + cifar_results[1,:,0]], axis=0)
    plt.bar(xaxis,
            # [np.mean(mnist_results[0,:,3] + mnist_results[0,:,0]),
            #  np.mean(cifar_results[0,:,3] + cifar_results[0,:,0])],
            np.mean(dram_data, axis=-1),
            bar_width, alpha=opacity, label="dram")
    plt.bar(xaxis+bar_width,
            # [np.mean(mnist_results[1,:,3] + mnist_results[1,:,0]),
            #  np.mean(cifar_results[1,:,3] + cifar_results[1,:,0])],
            np.mean(pymm_data, axis=-1),
            bar_width, alpha=opacity, label="pymm")
    plt.ylabel("wavelet construction + data loading time (s)")
    plt.xticks(xaxis+bar_width, ["mnist", "cifar10"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # total script time
    # print(np.sum(mnist_results[0,:,:], axis=-1).shape)
    dram_data = np.stack([mnist_results[0,:,:].sum(axis=-1),
                          cifar_results[0,:,:].sum(axis=-1)], axis=0)
    pymm_data = np.stack([mnist_results[1,:,:].sum(axis=-1),
                          cifar_results[1,:,:].sum(axis=-1)], axis=0)
    plt.bar(xaxis,
            # [np.mean(np.sum(mnist_results[0,:,:], axis=-1)),
            #          np.mean(np.sum(cifar_results[0,:,:], axis=-1))],
            np.mean(dram_data, axis=-1),
            bar_width, alpha=opacity, label="dram")
    plt.bar(xaxis+bar_width,
            # [np.mean(np.sum(mnist_results[1,:,:], axis=-1)),
            #          np.mean(np.sum(cifar_results[1,:,:], axis=-1))],
            np.mean(pymm_data, axis=-1),
            bar_width, alpha=opacity, label="pymm")
    plt.ylabel("total script time (s)")
    plt.xticks(xaxis+bar_width, ["mnist", "cifar10"])
    plt.legend()
    plt.show()
    ###########################################


if __name__ == "__main__":
    main()

