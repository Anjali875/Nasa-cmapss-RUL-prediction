# Predicting Engine Failure: NASA Turbofan (C-MAPSS FD001)

Predicting how many flights a jet engine has left before it fails, using sensor data.

This is called **Remaining Useful Life (RUL)** prediction. If you know an engine is close to failing, you can fix it on your schedule instead of after something breaks. This kind of predictive maintenance is used in aviation, manufacturing, and energy.

## Dataset

[NASA C-MAPSS](https://www.kaggle.com/) turbofan engine degradation simulation, subset **FD001**.

| File | What it has |
|---|---|
| `train_FD001.txt` | 100 engines, each run from healthy until failure (~20k rows) |
| `test_FD001.txt` | 100 other engines, cut off *before* failure |
| `RUL_FD001.txt` | The true remaining life of each test engine at its cutoff |

Each row is one flight (cycle) of one engine, with 3 operating settings and 21 sensor readings (temperatures, pressures, fan speeds, etc.).

## Approach

1. **EDA:** check engine life lengths, find flat sensors, plot sensor trends, look at noise and correlations.
2. **Target:** RUL = (last cycle of the engine) - (current cycle), capped at 125. Early in an engine's life there's no visible wear, so very high RUL values aren't learnable.
3. **Features:** drop flat sensors, then add rolling mean/std (5 and 10 cycles) and deltas, computed separately for each engine.
4. **Validation:** split by **engine ID**, not by random rows. Rows from the same engine are very similar, so a random split would leak information and give fake-good scores.
5. **Models:** Linear Regression as a baseline, then LightGBM.
6. **Alert rule:** flag an engine for maintenance when predicted RUL < 30 cycles, and measure precision and recall.

## Project structure

```
nasa-turbofan-ml/
├── data/                       # dataset files (not in the repo)
├── cmapss_fd001_part1.ipynb    # EDA, target, features, engine-level split
├── README.md
└── .gitignore
```

## How to run

1. Clone the repo.
2. Download the C-MAPSS data from Kaggle and put `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in a `data/` folder.
3. Create an environment and install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac/Linux
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm ipykernel
   ```
4. Open the notebook, select the `.venv` kernel, and run the cells from top to bottom. Set `DATA_DIR = "data"` in the loading cell.

## Tech

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, seaborn
