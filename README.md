# qrm_optimal_execution

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/espanato/qrm_optimal_execution.git
cd qrm_optimal_execution
```

### 2. Create the virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the `scripts\run_train.py` file and have fun!



### 📌 \***\*Last update (29/05):\*\*** uploaded `simulate_qrm_numba.py`, `rl_utils_numba.py` and updated the folder `calibration_data`

---

Updated files for the QRM simulation and the RL environment.

- `simulate_qrm_numba.py`: numba decorated functions to speed up the QRM simulation.
- `rl_utils_numba.py`: full numpy version of the LOB to avoid Python overhead. Implementation of the _RLRunner_ class to train and test. Implementation of the benchmark strategies _TWAPAgent_, _FrontLoadAgent_ and _BackLoadAgent_.

### 📌 \***\*Update (16/05):\*\*** uploaded `simulate_qrm.py`, `rl_utils.py` and the folder `calibration_data`

---

Python files for the QRM simulation and first version of the RL environment.

- `simulate_qrm.py` gathers the functions to simulate the QRM model of Huang et al. (2015) [![arXiv](https://img.shields.io/badge/arXiv-1312.0563-b31b1b.svg)](https://arxiv.org/abs/1312.0563).
- `rl_utils.py` gathers the functions for the RL environment with DDQN Agent.
- `calibration_data` is a folder of data for the invariant distributions and intensity tables.

### 📌 \***\*Update (02/05):\*\*** uploaded the file `main_git.ipynb`

---

First version of the QRM code and initial simulations. This notebook:

- Creates an animation of the Limit Order Book (LOB) using the same parameters as the QRM paper.
- If you run the entire notebook:
  - Saves a `.npy` file containing the invariant distribution.
  - Generates a `.gif` of the LOB animation, including mid-price evolution (typically showing 4–5 mid-price changes).
