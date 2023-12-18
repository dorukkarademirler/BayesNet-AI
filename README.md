# README

## Overview

This assignment focuses on implementing inference algorithms for Directed Acyclic Graphs (DAGs), specifically using Variable Elimination and Likelihood Sampling. Additionally, it explores the application of these algorithms in the context of Causal Bayesian Networks (BNs), using real COVID-19 data.

## Files

- `solution.py`: Contains the implemented functions for Variable Elimination, Likelihood Sampling, and Causal Bayesian Network creation.
- `bnetbase.py`: Provides class definitions for Variable, Factor, and BN objects.
- `covid.csv`: Dataset containing COVID-19 data for Italy and China.
- `bns.pdf`: Diagram illustrating causal relationships for a hypothetical diabetes-related dataset.

## Instructions

### Part 1: Exact Inference with Variable Elimination

- Implemented the `VE` function to perform exact inference using Variable Elimination.
- Implemented helper functions: `multiply_factors` and `min_fill_ordering`.

### Part 2: Approximate Inference with Likelihood Sampling

- Implemented the `SampleBN` function to perform approximate inference using Likelihood Sampling.

### Part 3: Causal Graphs

- Implemented `CausalModelMediator` and `CausalModelConfounder` methods to create Causal Bayesian Networks.

### Part 4: Estimating Causal Effects

- Implemented additional helper functions as needed.
- Estimated the causal effect of Country (Italy v. China) on Fatality in both networks.

## Running the Code

Ensure you have Python installed. Run the `solution.py` file to execute the implemented functions.

```bash
python solution.py
