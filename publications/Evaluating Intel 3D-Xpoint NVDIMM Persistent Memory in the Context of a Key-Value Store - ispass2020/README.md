# Evaluating Intel 3D-Xpoint NVDIMM Persistent Memory in the Context of a Key-Value Store 
## Abstract
Intel's 3D-Xpoint NVDIMM product1 is now in general availability. This technology, herein termed Optane-PM, is a first-in-breed persistent memory technology, designed for the enterprise storage domain. Optane-PM is attached to the memory-bus and is load-store addressable. It provides higher capacity and density than DRAM, while also enabling non-volatility - data is persistent and durable across power-cycles. This paper presents a detailed evaluation of adopting Optane-PM in key-value stores. To achieve this goal, we designed and implemented a high-performance, memory-centric key-value store (MCKVS), that directly leverages persistent memory for data and meta-data storage. Multiple storage back-ends, with different software stacks, are used to provide a comparative analysis and help us understand how Optane-PM is positioned on the performance landscape. We also conducted comparisons against popular in-memory KV stores based on DRAM.

## Link to the paper: 
https://ieeexplore.ieee.org/document/9238605

## bibtex entry:
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
