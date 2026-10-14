# Machine Learning-Based EV Battery Management System

A machine learning project that combines **EV battery capacity estimation, Remaining Useful Life (RUL) prediction, and reinforcement-learning-based charging strategy evaluation** in a Streamlit application.

## Overview

This project explores how machine learning can support battery condition monitoring and how different charging strategies behave in a simulated environment.

The system uses battery cycling data from NASA's Prognostics Center of Excellence (PCoE) dataset to build supervised learning models. It also compares fixed-current, adaptive, and Proximal Policy Optimization (PPO) charging strategies using a simplified charging simulator.

### Key Features

* **Battery Capacity Estimation:** Uses an Extra Trees regression model to estimate discharge capacity.
* **Remaining Useful Life Prediction:** Uses a Random Forest regression model to estimate remaining cycles before a defined end-of-life threshold.
* **Charging Strategy Comparison:** Evaluates fixed-current, adaptive, and PPO-based charging strategies.
* **Performance Visualization:** Compares charging duration, simulated temperature, and a model-defined stress indicator.
* **Streamlit Interface:** Provides an interactive interface for presenting predictions and charging simulation results.

## System Architecture

The project consists of the following modules:

1. **Data Processing:** Processes NASA battery cycling records into cycle-level features.
2. **Capacity Prediction:** Estimates battery discharge capacity using Extra Trees regression.
3. **RUL Prediction:** Estimates remaining cycles using Random Forest regression.
4. **Charging Simulation:** Compares conventional and reinforcement-learning-based charging policies.
5. **Visualization and Interface:** Presents model outputs and experimental comparisons through Streamlit.

## Dataset

The project uses four battery cells from the NASA PCoE battery dataset:

* B0005
* B0006
* B0007
* B0018

The processed feature table contains **636 records and 20 columns**. Features summarize cycle-level voltage, current, temperature, time, and battery-related variables.

The reference capacity is set to 2.0 Ah, and the selected end-of-life threshold is approximately 1.4 Ah (70% of reference capacity).

**Dataset source:** [NASA Prognostics Center of Excellence — Battery Dataset](https://c3.ndc.nasa.gov/dashlink/resources/133/)

## Technologies Used

* Python
* pandas and NumPy
* scikit-learn
* Extra Trees and Random Forest regression
* Reinforcement Learning with Proximal Policy Optimization (PPO)
* Stable-Baselines3 (for PPO, if used in the implementation)
* Streamlit
* Matplotlib
* Jupyter Notebook / Google Colab

## Reported Results

### Capacity Prediction

The Extra Trees model's reported leave-one-battery-out evaluation results are:

| Held-out battery  |   MAE (Ah) |  RMSE (Ah) |
| ----------------- | ---------: | ---------: |
| B0005             |     0.0246 |     0.0289 |
| B0006             |     0.0455 |     0.0640 |
| B0007             |     0.0256 |     0.0301 |
| B0018             |     0.0301 |     0.0331 |
| **Reported mean** | **0.0314** | **0.0390** |

### Remaining Useful Life Prediction

| Metric            |          Reported result |
| ----------------- | -----------------------: |
| Model             | Random Forest Regression |
| Evaluated records |                      468 |
| MAE               |              6.84 cycles |
| RMSE              |              9.23 cycles |

B0007 did not cross the selected end-of-life threshold during the available observations, so valid RUL labels could not be constructed for that cell under the chosen rule.

### Charging Strategy Comparison

Results from the reported 100-randomized-scenario experiment:

| Strategy      | Mean charging time | Mean maximum simulated temperature | Mean stress indicator |
| ------------- | -----------------: | ---------------------------------: | --------------------: |
| Fixed-current |          49.99 min |                          29.454 °C |                 1.001 |
| Adaptive      |          86.11 min |                          28.649 °C |                 0.635 |
| PPO V3        |          55.33 min |                          29.149 °C |                 0.918 |

All three strategies achieved 100% target completion in the reported simulation run.

The fixed-current strategy had the shortest mean charging duration, while the adaptive strategy had the lowest mean simulated temperature and stress indicator. These are simulation results, not measurements of real battery degradation or proof of physical charging safety.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Priya9407/ev-battery-management-system.git
cd ev-battery-management-system
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages required by the implementation. For example:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

If the charging module uses Stable-Baselines3, Gymnasium, or PyTorch, install the compatible versions required by that module as well.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Replace `app.py` with the actual Streamlit entry-point filename if it differs.

## Project Structure

The exact repository structure may differ. A suggested organization is:

```text
ev-battery-management-system/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── battery_discharge_features.csv
├── models/
│   ├── capacity_prediction_model.joblib
│   └── ppo_ev_charging_v3.zip
├── notebooks/
├── results/
│   ├── charging_strategy_comparison.csv
│   └── randomized_100_scenario_results.csv
└── figures/
    ├── charging_time_comparison.png
    ├── charging_temperature_comparison.png
    └── charging_stress_comparison.png
```

This is an illustrative structure; filenames and directories should be adjusted to match the actual repository.

## Limitations

* The supervised learning dataset contains only four battery cells, limiting conclusions about generalization to other batteries.
* RUL estimates depend on the selected end-of-life threshold and the availability of observed threshold crossings.
* The charging simulator uses simplified state-of-charge, thermal, and stress equations.
* The stress indicator is a simulation-defined proxy, not a validated physical degradation measurement.
* The system is an academic prototype and is not a safety-certified battery management controller.

## Future Work

* Evaluate models on additional battery cells and operating conditions.
* Audit feature selection to prevent target leakage.
* Improve charging simulation using experimentally calibrated battery dynamics.
* Evaluate charging policies using repeated, identical scenario sets.
* Add prediction uncertainty, stronger input validation, and model-version tracking.

## Contributors

* Priyadharshini V
* Subhashree M
* Architaa A

Department of Computer Science and Engineering
Vellore Institute of Technology, Chennai

## Disclaimer

This project is intended for educational and research purposes. Predictions and charging comparisons are experimental outputs and must not be used as the sole basis for real-world EV battery operation or safety decisions.

