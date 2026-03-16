# PFE: Continuous-Time Stochastic Optimal Control

**Student:** Youssef Kallala  
**Supervisors:**  
* R. Tempone  
* Erik von Schwerin  
* Sebastian Leonardo Lalvay Segovia 

**Project:** Hamiltonian-Partials Reformulation + Pathwise Discrete Dual Gradients  

---

## Overview

This is a Bachelor/PFE (Projet de fin d'études) capstone project.

The goal is to implement a numerical framework that turns stochastic policy search into unconstrained optimization over a smooth potential function $\bar{u}(t,x)$.

The project combines:
* Continuous-time stochastic control
* Hamiltonian reformulations
* Pathwise discrete dual gradient methods
* Analytical validation on LQR / LQG benchmarks

---

## Repository Structure

```text
soc/        Core package (Models, Hamiltonian, Simulation, Dual Gradients)
docs/       LaTeX documents (Notation page, final thesis)
scripts/    Entry points for running experiments and suites
tests/      Unit tests for analytic benchmarks (LQR/LQG)
configs/    Configuration files for reproducibility
figures/    Auto-generated plots for the thesis
results/    Saved metrics and RNG states
```

---

## 🧮 Mathematical Notation & Codebase Mapping

To ensure strict consistency between the mathematical formulation and the Python implementation, all variable names and symbols are frozen.

Please see the [Notation Page](docs/notation.pdf) for the complete list of symbols, dimensions, and their exact corresponding Python variables in the codebase.

---

## Getting Started

### 1️⃣ Install dependencies

Make sure you have Python 3 installed. Install the required packages by running:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run unit tests (Math Verification)

To verify that the underlying math (e.g., matrix symmetry, LQR/LQG analytic costs) is correct across all integration methods (Euler, RK2, RK4), run the automated test suite:

```bash
python -m pytest -v tests/
```
*(Note: Using `python -m pytest` ensures Python correctly finds the `soc` package in your directory without import errors).*

### 3️⃣ Run Experiment 0 (Deterministic LQR)

To run the deterministic LQR sanity check and generate the visual convergence plots, run the following script:

```bash
python scripts/exp0_lqr_sanity.py
```
*This will output the convergence errors to the terminal, display the graph, and save the resulting `.png` file directly into the `figures/` folder for use in the thesis.*

### 4️⃣ Run Experiment 1 (Stochastic LQG Benchmark)

To run the stochastic LQG benchmark using the Euler-Maruyama scheme and generate Monte Carlo statistical convergence plots, run:

```bash
python scripts/run_exp1_lqg.py
```
*This script simulates parallel SDE trajectories, computes the expected Monte Carlo costs using Student's t-distribution (95% Confidence Intervals), compares them to the exact analytic solution, and saves the thesis-ready figure to the `figures/` folder.*

---


## Author

**Youssef Kallala**  
Bachelor Thesis / PFE  
March 2026
