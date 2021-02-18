# Code files for PhD dissertation of Adam Glos

Person responsible for data: *Adam Glos* (aglos [at] iitis.pl).

The repository contains script files for generating and plotting data used in PhD dissertation.

## Used software
In order to run our scripts, Julia v. 1.5.2 is required and Anaconda 4.9.2 (the code was not tested on older versions). The data was generated using Ubuntu OS, and was not tested on other OS.

## Data

In order to run the Julia code, one has to run `julia` and type `]activate .`, then run `instantiate` to load the environment. The following`.jl` files 
* propagation\_qsw.jl
* nonmoralizing\_qsw\_plotter.jl
* convergence\_qsw\_prob.jl
* convergence\_qsw\_states.jl
* convergence\_nonrelaxing.jl
* complex\_search.jl
* hiding\_search.jl

are responsible for generating the data. It is enough to run `julia file.jl` in the terminal. 

## Plotting

For generating plots one has to set up and activate conda environment through 
```
conda env create -f rozprawa.yml
conda activate rozprawa
```
Files _*plotter.jl_ are responsible for generating the data. One can run them through `julia file.jl`.
