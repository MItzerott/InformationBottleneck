# InformationBottleneck
Optimized version of the code published in Strouse and Schwab (2017) under https://github.com/djstrouse/information-bottleneck.git (see their documentation)

Several implementations have been made to adress the following limitations:

- The reference implementation is tailored to a small two-dimensional toy dataset.
- Boundary effects are not explicitly treated.
- Computational resource utilization is not optimal for large datasets.
- No data storage functionality is provided.
- Deterministic annealing strategies are not implemented.
- Numerical Instabilities occur in the evaluation of the KL divergence at large $\beta$ values.
- Configuration parameters are hard-coded in the source code rather than defined via an external configuration file.
    
to be able to operate the code on more complex datasets (in this case specifically but not limited to cosmological simulations)


Parts of the code were rewritten in Cython to accomodate larger numerical calculations, specifically for calculating the Kullback Leibler divergence. OpenMP is used to parallelize the calculation. The code can now generally be used on any dataset with any dimensionality. 
