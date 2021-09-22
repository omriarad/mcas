# List of publications

1. Non-Volatile Memory Accelerated Geometric Multi-Scale Resolution Analysis - HPEC 2021 (see code example [GMRA](./code_examples/GMRA_HPEC2021))

2. Non-Volatile Memory Accelerated Posterior Estimation - HPEC 2021 (see code example [Postrerior_estimation](./code_examples/Posterior_estimation_HPEC2021))

3. An Architecture for Memory Centric Active Storage (MCAS) - arxiv

4. A High-Performance Persistent Memory Key-Value Store with Near-Memory Compute - arxiv

5. Evaluating Intel 3D-Xpoint NVDIMM Persistent Memory in the Context of a Key-Value Store - ISPASS 2020 
 

# More Details  


## Non-Volatile Memory Accelerated Geometric Multi-Scale Resolution Analysis
### Abstract
mensionality reduction algorithms are standard tools in a researcher’s toolbox. Dimensionality reduction algorithms are frequently used to augment downstream tasks such as machine learning, data science, and also are exploratory methods for understanding complex phenomena. For instance, dimensionality reduction is commonly used in Biology as well as Neuroscience to understand data collected from biological subjects. However, dimensionality reduction techniques are limited by the von-Neumann architectures that they execute on. Specifically, data intensive algorithms such as dimensionality
reduction techniques often require fast, high capacity, persistent memory which historically hardware has been unable to provide at the same time. In this paper, we present a re-implementation of an existing dimensionality reduction technique called Geometric Multi-Scale Resolution Analysis (GMRA) which has been accelerated via novel persistent memory technology called Memory Centric Active Storage (MCAS). Our implementation uses a specialized version of MCAS called PyMM that provides native support for Python datatypes including NumPy arrays and PyTorch tensors. We compare our PyMM implementation against a DRAM implementation, and show that when data fits in DRAM, PyMM offers competitive runtimes. When data does not fit in DRAM, our PyMM implementation is still able to process the data.

### Link for the paper: 

### Bibtex entry


## Non-Volatile Memory Accelerated Posterior Estimation
### Abstract

Bayesian inference allows machine learning models to express uncertainty. Current machine learning models use only a single learnable parameter combination when making predictions, and as a result are highly overconfident when their predictions are wrong. To use more learnable parameter combinations efficiently, these samples must be drawn from the posterior distribution. Unfortunately computing the posterior directly is infeasible, so often researchers approximate it with a well known distribution such as a Gaussian. In this paper, we show that through the use of high-capacity persistent storage, models whose posterior distribution was too big to approximate are now feasible, leading to improved predictions in downstream tasks.



### Link to the paper:

### Bibtex






## An Architecture for Memory Centric Active Storage (MCAS) - arxiv
### Abstract
The advent of CPU-attached persistent memory technology, such as Intel's Optane Persistent Memory Modules (PMM),
has brought with it new opportunities for storage. In 2018, IBM Research Almaden began investigating and developing
a new enterprise-grade storage solution directly aimed at this emerging technology.
MCAS (Memory Centric Active Storage) defines an "evolved" network-attached key-value store that offers both near-data compute and the ability to layer enterprise-grade data management services on shared persistent memory. As a converged memory-storage tier, MCAS moves towards eliminating he traditional separation of compute and storage, and thereby unifying the data space. This paper provides an in-depth review of the MCAS architecture
and implementation, as well as general performance
results.

### Link to the paper:
 https://arxiv.org/abs/2103.00007

### Bibtex entry:
```
@misc{waddington2021architecture,
      title={An Architecture for Memory Centric Active Storage (MCAS)}, 
      author={Daniel Waddington and Clem Dickey and Moshik Hershcovitch and Sangeetha Seshadri},
      year={2021},
      eprint={2103.00007},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```

## A High-Performance Persistent Memory Key-Value Store with Near-Memory Compute - arxiv

### Abstract

MCAS (Memory Centric Active Storage) is a persistent memory tier for high-performance durable data storage. It is designed from the ground-up to provide a key-value capability with low-latency guarantees and data durability through memory persistence and replication. To reduce data movement and make further gains in performance, we provide support for user-defined "push-down" operations (known as Active Data Objects) that can execute directly and safely on the value-memory associated with one or more keys. The ADO mechanism allows complex pointer-based dynamic data structures (e.g., trees) to be stored and operated on in persistent memory. To this end, we examine a real-world use case for MCAS-ADO in the handling of enterprise storage system metadata for Continuous Data Protection (CDP). This requires continuously updated complex metadata that must be kept consistent and durable.
In this paper, we i.) present the MCAS-ADO system architecture, ii.) show how the CDP use case is implemented, and finally iii.) give an evaluation of system performance in the context of this use case.

### Link to the paper: 
https://arxiv.org/abs/2104.06225

### Bibtex entry
```
@misc{waddington2021highperformance,
      title={A High-Performance Persistent Memory Key-Value Store with Near-Memory Compute}, 
      author={Daniel Waddington and Clem Dickey and Luna Xu and Moshik Hershcovitch and Sangeetha Seshadri},
      year={2021},
      eprint={2104.06225},
      archivePrefix={arXiv},
      primaryClass={cs.DB}
}
```

## Evaluating Intel 3D-Xpoint NVDIMM Persistent Memory in the Context of a Key-Value Store 
### Abstract
Intel's 3D-Xpoint NVDIMM product1 is now in general availability. This technology, herein termed Optane-PM, is a first-in-breed persistent memory technology, designed for the enterprise storage domain. Optane-PM is attached to the memory-bus and is load-store addressable. It provides higher capacity and density than DRAM, while also enabling non-volatility - data is persistent and durable across power-cycles. This paper presents a detailed evaluation of adopting Optane-PM in key-value stores. To achieve this goal, we designed and implemented a high-performance, memory-centric key-value store (MCKVS), that directly leverages persistent memory for data and meta-data storage. Multiple storage back-ends, with different software stacks, are used to provide a comparative analysis and help us understand how Optane-PM is positioned on the performance landscape. We also conducted comparisons against popular in-memory KV stores based on DRAM.

### Link to the paper: 
https://ieeexplore.ieee.org/document/9238605

### Bibtex entry:
```
@INPROCEEDINGS{9238605,
  author={Waddington, Daniel and Dickey, Clem and Xu, Luna and Janssen, Travis and Tran, Jantz and Kshitij, Doshi},
  booktitle={2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)}, 
  title={Evaluating Intel 3D-Xpoint NVDIMM Persistent Memory in the Context of a Key-Value Store}, 
  year={2020},
  volume={},
  number={},
  pages={202-211},
  doi={10.1109/ISPASS48437.2020.00035}}
```
