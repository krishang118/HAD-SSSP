# HAD-SSSP: Hierarchical Adaptive Distributed Single-Source Shortest Path

A novel high-performance distributed graph algorithm that combines multi-level hierarchical clustering with asynchronous message-passing to solve the Single-Source Shortest Path (SSSP) problem efficiently across diverse graph topologies.

## Features

- Hierarchical Clustering: Four-level spatial hierarchy (L0-L3) for fast approximate distance queries
- Graph-Adaptive: Automatically tunes parameters for road networks, scale-free graphs, and general topologies
- Hybrid Parallelism: MPI for distributed computing + OpenMP for multi-threaded edge relaxation
- Superior Performance: Up to 6.75× faster than state-of-the-art algorithms on scale-free graphs
- Strong Scalability: Near-linear scaling from 4 to 16 MPI processes

## Algorithm Architecture

### Four-Level Hierarchy

- L0 (Base): Original graph nodes
- L1 (Local): BFS-based local clusters (20-50 nodes)
- L2 (Regional): Spatial hash clustering of L1 clusters
- L3 (Supernodes): Coarse-grained aggregation for O(1) distance approximation

### Owner-Push Message Passing

- Each MPI rank owns a contiguous range of nodes
- Asynchronous distributed Dijkstra with bounded message complexity
- Batch message aggregation reduces MPI overhead
- Parallel edge relaxation for high-degree nodes (OpenMP)

### Supported datasets (automatically detected):

- `random_citation_graph.txt` - Citation Graph (100K nodes, 250K edges)
- `roadNet-CA.txt` - California Road Network (1.9M nodes, 2.7M edges)
- `graph500-scale20-ef16_adjedges.txt` - Graph500 (1M nodes, 16.8M edges)

## How to Run

1. Clone this repository on your local machine.
2. Run and execute the `HAD-SSSP.cpp` file.

## Research Paper

For detailed methodology and mathematical formulations, you may refer to the uploaded research paper `Research Paper.pdf`.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License.
