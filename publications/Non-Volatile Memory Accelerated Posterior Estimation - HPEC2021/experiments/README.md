This is our experiments directory. There are two main scripts at the moment (am still working on memory mapping on DRAM script). These scripts are:

1) ./mnist_dram.py
2) ./mnist_pymm.py

These scripts will run a SWAG (SWA-Gaussian) experiment using either pymm or dram. Both scripts take a few optional arguments and only one required argument. The required argument is the model string (aka class name of the model in lower-case).

The posterior in these experiments is a function of the number of samples. The number of bytes occupied by the posterior follows this rule:

            Bytes(Posterior) = 12*n*(k+3) + 12

where n is the number of parameters in the model, and k is the number of samples that the posterior expects (following the original SWAG paper https://proceedings.neurips.cc/paper/2019/file/118921efba23fc329e6560b27861f0c2-Paper.pdf).

The following models are available (and are the required argument for each script to run):
1) model_2conv2fc. This model has 2 conv layers and 2 fc layers totaling 21,840 params
2) model_2fc. This model has 2 fc layers totaling 795,010 params
3) model_3fc. This model has 3 fc layers totaling 1,290,510 params
4) model_4fc. This model has 4 fc layers totaling 2,797,010 params

The default behavior for the scripts is to use all posterior samples (recorded after every minibatch as default behavior). For MNIST, there are 600 batches using the standard minibatch size of 100). Given the batch_size and the number of epochs, and the recording frequency (i.e. 1/num_minibatches_to_record_a_sample), we can compute the parameter k to be:

            k = 60000 * epochs * frequency / batch_size

Using default values, if we wanted to target the following posterior memory sizes, we would need to train for the following number of epochs:

model               1gb             10gb            100gb           300gb           700gb
------------------------------------------------------------------------------------------
model_2conv2fc      7               64              636             1908            4452
model_2fc           1               2               18              53              123
model_3fc           1               2               11              33              76
model_4fc           1               1               5               15              35


By defualt, results will be saved to ./results/mnist/{dram,pymm}.csv unless a path is specified otherwise. Please use the -h flag to see the arguments for the scripts.

