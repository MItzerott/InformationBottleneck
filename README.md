# InformationBottleneck
Optimized version of the code published in Strouse and Schwab (2017) under https://github.com/djstrouse/information-bottleneck.git

Several implementations have been made to adress the following limitations:

    \item The reference implementation is tailored to a small two-dimensional toy dataset (see \cite{strouse2020information}).
    \item Boundary effects are not explicitly treated.
    \item Computational resource utilization is not optimal for large datasets.
    \item No data storage functionality is provided.
    \item Deterministic annealing strategies (see iIB in section \ref{sec:theory_theIBMethod_adaptations_singleSided}) are not implemented.
    \item Numerical Instabilities occur in the evaluation of the KL divergence (\autoref{equ:KL_DIV}) at large $\beta$ values.
    \item Configuration parameters are hard-coded in the source code rather than defined via an external configuration file.
    
to be able to operate the code on more complex datasets (in this case specifically but not limited to cosmological simulations)

 
