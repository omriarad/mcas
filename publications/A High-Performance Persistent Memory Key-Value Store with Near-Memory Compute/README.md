# A High-Performance Persistent Memory Key-Value Store with Near-Memory Compute

## Abstract

MCAS (Memory Centric Active Storage) is a persistent memory tier for high-performance durable data storage. It is designed from the ground-up to provide a key-value capability with low-latency guarantees and data durability through memory persistence and replication. To reduce data movement and make further gains in performance, we provide support for user-defined "push-down" operations (known as Active Data Objects) that can execute directly and safely on the value-memory associated with one or more keys. The ADO mechanism allows complex pointer-based dynamic data structures (e.g., trees) to be stored and operated on in persistent memory. To this end, we examine a real-world use case for MCAS-ADO in the handling of enterprise storage system metadata for Continuous Data Protection (CDP). This requires continuously updated complex metadata that must be kept consistent and durable.
In this paper, we i.) present the MCAS-ADO system architecture, ii.) show how the CDP use case is implemented, and finally iii.) give an evaluation of system performance in the context of this use case.

## Link to the paper: 
https://arxiv.org/abs/2104.06225

## bibtex entry:
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
