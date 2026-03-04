# PFE: Continuous-Time Stochastic Optimal Control

**Student:** Youssef Kallala
**Supervisors:**

* R. Tempone
* Erik von Schwerin
* Sebastian Leonardo Lalvay Segovia (PhD Student)

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

```bash
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

```bash
pip install -r requirements.txt
```

### 2️⃣ Run tests

```bash
pytest tests/
```

---

## Project Objectives

* Reformulate stochastic optimal control using Hamiltonian partials
* Implement pathwise dual gradient estimators
* Benchmark against analytical LQR/LQG solutions
* Provide reproducible experiments with configuration tracking
* Generate thesis-ready figures automatically

---

## Author

**Y. Kallala**
Bachelor Thesis / PFE
Continuous-Time Stochastic Optimal Control
