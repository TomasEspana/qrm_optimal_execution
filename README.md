# RL for Optimal Execution in Queue-Reactive Models

Code used in our paper: [![arXiv](https://img.shields.io/badge/arXiv-2511.15262-b31b1b.svg)](https://arxiv.org/abs/2511.15262)

## 📄 Abstract

We investigate the use of Reinforcement Learning for the optimal execution of meta-orders, where the objective is to execute incrementally large orders while minimizing implementation shortfall and market impact over an extended period of time. Departing from traditional parametric approaches to price dynamics and impact modeling, we adopt a model-free, data-driven framework. Since policy optimization requires counterfactual feedback that historical data cannot provide, we employ the Queue-Reactive Model to generate realistic and tractable limit order book simulations that encompass transient price impact, and nonlinear and dynamic order flow responses. Methodologically, we train a Double Deep Q-Network agent on a state space comprising time, inventory, price, and depth variables, and evaluate its performance against established benchmarks. Numerical simulation results show that the agent learns a policy that is both strategic and tactical, adapting effectively to order book conditions and outperforming standard approaches across multiple training configurations. These findings provide strong evidence that model-free Reinforcement Learning can yield adaptive and robust solutions to the optimal execution problem.

## 🚀 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/espanato/qrm_optimal_execution.git
cd qrm_optimal_execution
```

### 2. Virtual env and dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. See the quick guide below and have fun!


## 🚀 The Queue-Reactive Model 

The Queue-Reactive Model corresponds to Model I of the paper: [![arXiv](https://img.shields.io/badge/arXiv-1312.0563-b31b1b.svg)](https://arxiv.org/abs/1312.0563)