```markdown
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
scripts/    Entry points for running experiments and suites
tests/      Unit tests for analytic benchmarks (LQR/LQG)
configs/    Configuration files for reproducibility
figures/    Auto-generated plots for the thesis
results/    Saved metrics and RNG states
```

---

## Getting Started

### 1️⃣ Install dependencies

Make sure you have Python 3 installed. Install the required packages by running:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run unit tests (Math Verification)

To verify that the underlying math (e.g., matrix symmetry, LQR analytic costs) is correct across all integration methods (Euler, RK2, RK4), run the automated test suite:

```bash
python -m pytest tests/
```
*(Note: Using `python -m pytest` ensures Python correctly finds the `soc` package in your directory without import errors).*

### 3️⃣ Run Experiment 0 (Generate Plots)

To run the deterministic LQR sanity check and generate the visual convergence plots, run the following script:

```bash
python scripts/exp0_lqr_sanity.py
```
*This will output the convergence errors to the terminal, display the graph, and save the resulting `.png` file directly into the `figures/` folder for use in the thesis.*

---

## Project Objectives

* Reformulate stochastic optimal control using Hamiltonian partials
* Implement pathwise dual gradient estimators
* Benchmark against analytical LQR/LQG solutions
* Provide reproducible experiments with configuration tracking
* Generate thesis-ready figures automatically

---

## Author

**Youssef Kallala**  
Bachelor Thesis / PFE  
Continuous-Time Stochastic Optimal Control  
```