# Info345ol
Group Assignment for INFO345 by Gaute Robertsen, Sondre Angeltveit and Eivind Englokk

Recommender systems on the RateBeer dataset (Lederrey & West, 2018).

## Files
| File | What it does |
|---|---|
| `00_prepare_data.py` | Step 0: cuts the full `ratings.csv` down to ratings from 2012–2017 and writes summary statistics |
| `01_data_analysis.ipynb` | Task 1: exploratory data analysis and filtering |
| `02_models_and_evaluation.ipynb` | Task 2 & 3: Surprise models (SVD, SVD++, KNN-Basic, KNN-Baseline, Baseline, Random), 5-fold cross validation, RMSE & MAE |
| `figures/` | Figures for the report |

## Setup
The dataset is **not** in this repo (too large). The code expects this layout:
```
RateBeer Dataset/            <- ratings.csv, beers.csv, breweries.csv, users.csv
  Info345ol/                 <- this repo (clone it inside the dataset folder)
```
So: `cd "RateBeer Dataset"` and then `git clone https://github.com/GauteRobertsen/Info345ol.git`.

Python packages: `pandas pyarrow numpy matplotlib scikit-surprise tabulate jupyter`

## How to run
1. `python 00_prepare_data.py` (a few minutes; creates `ratings_2012_2017.parquet` and `stats_full/` in the dataset folder)
2. Run `01_data_analysis.ipynb` (creates `filtered.parquet` in the dataset folder)
3. Run `02_models_and_evaluation.ipynb` (about 15 minutes; SVD++ is the slowest)

To keep SVD++ trainable on a laptop, notebook 2 uses at most 100 random ratings per user (592,205 ratings, all 10,368 users).

## Results (5-fold cross validation)
| Recommenders | Pred Accuracy [RMSE] | Pred Accuracy [MAE] |
|:--|--:|--:|
| SVD | 0.533 | 0.392 |
| SVD++ | 0.520 | 0.381 |
| KNN-Basic | 0.574 | 0.419 |
| KNN-Baseline | 0.533 | 0.390 |
| Baseline | 0.532 | 0.390 |
| Random | 1.001 | 0.790 |

## Data source
Lederrey, G. & West, R. (2018). When Sheep Shop: Measuring Herding Effects in Product Ratings with Natural Experiments. *WWW 2018*.
