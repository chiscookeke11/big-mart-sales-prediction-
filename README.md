# Big Mart Sales Prediction

A notebook-based regression project that predicts **`Item_Outlet_Sales`** for retail products sold through Big Mart outlets. The project uses exploratory data analysis, simple missing-value imputation, categorical label encoding, and an XGBoost regressor to estimate outlet sales from item and store attributes.

> **Project status:** the included notebook trains and evaluates a model on a holdout split of the supplied training data. It does not currently generate or save predictions for `sample_data/Test.csv`.

## Contents

- [Project workflow](#project-workflow)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Setup](#setup)
- [Run the notebook](#run-the-notebook)
- [Modeling approach](#modeling-approach)
- [Results](#results)
- [Reproducing the workflow in Python](#reproducing-the-workflow-in-python)
- [Limitations and next steps](#limitations-and-next-steps)

## Project workflow

The notebook (`big_mart_sales_prediction.ipynb`) follows this sequence:

1. Loads the Big Mart training file from `sample_data/Train.csv`.
2. Inspects dataset dimensions, descriptive statistics, data types, and missing values.
3. Visualizes numerical distributions and categorical counts with Matplotlib and Seaborn.
4. Imputes missing item weights with the training-set mean.
5. Imputes missing outlet sizes with the modal size within each `Outlet_Type` group.
6. Standardizes inconsistent `Item_Fat_Content` labels such as `LF`, `low fat`, and `reg`.
7. Label-encodes the categorical features.
8. Separates `Item_Outlet_Sales` as the target, then makes an 80/20 train/test split using `random_state=2`.
9. Trains an `XGBRegressor` and reports R² scores for the train and holdout partitions.

## Repository layout

```text
.
├── big_mart_sales_prediction.ipynb  # EDA, preprocessing, training, evaluation
├── sample_data/
│   ├── Train.csv                    # Big Mart labeled training data
│   ├── Test.csv                     # Big Mart unlabeled prediction data
│   ├── sample_submission.csv        # Submission-format example
│   ├── train/train.csv              # Unrelated text-classification sample data
│   ├── val/val.csv                  # Unrelated text-classification sample data
│   └── test/test.csv                # Unrelated text-classification sample data
└── README.md
```

Only `sample_data/Train.csv` is read by the current notebook. The lowercase `train`, `val`, and `test` directories contain text/label data that are not part of the Big Mart workflow.

## Data

The Big Mart training data contains **8,523 rows** and **12 columns**. `Item_Outlet_Sales` is the regression target; the remaining 11 columns are model inputs.

| Column | Type | Description |
| --- | --- | --- |
| `Item_Identifier` | categorical | Product identifier. |
| `Item_Weight` | numeric | Product weight. |
| `Item_Fat_Content` | categorical | Product fat-content category. |
| `Item_Visibility` | numeric | Proportion of total display area allocated to the product. |
| `Item_Type` | categorical | Product category. |
| `Item_MRP` | numeric | Maximum retail price. |
| `Outlet_Identifier` | categorical | Store identifier. |
| `Outlet_Establishment_Year` | numeric | Year the store was established. |
| `Outlet_Size` | categorical | Store-size category. |
| `Outlet_Location_Type` | categorical | Store location tier. |
| `Outlet_Type` | categorical | Store format. |
| `Item_Outlet_Sales` | numeric | **Target:** sales of the product at the outlet. |

### Missing values handled by the notebook

| Feature | Missing entries | Treatment |
| --- | ---: | --- |
| `Item_Weight` | 1,463 | Fill with the column mean. |
| `Outlet_Size` | 2,410 | Fill with the mode for the corresponding `Outlet_Type`. |

## Setup

### Prerequisites

- Python 3.9 or later is recommended.
- Jupyter Notebook or JupyterLab.
- A C/C++-compatible environment supported by the installed `xgboost` wheel, if your platform requires it.

### Install dependencies

Create and activate a virtual environment if desired, then install the packages used by the notebook:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Run the notebook

From the repository root:

```bash
jupyter notebook big_mart_sales_prediction.ipynb
```

Or with JupyterLab:

```bash
jupyter lab big_mart_sales_prediction.ipynb
```

Run cells from top to bottom. The notebook uses the relative path `./sample_data/Train.csv`, so launch Jupyter from the repository root (or update the path before execution).

## Modeling approach

### Preprocessing

The notebook applies the following transformations before fitting the model:

- Converts `low fat` and `LF` to `Low Fat`, and `reg` to `Regular`.
- Uses `sklearn.preprocessing.LabelEncoder` to encode `Item_Identifier`, `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`.
- Leaves numerical columns in their original scale.

### Train/validation split

The data is split with:

```python
train_test_split(X, Y, test_size=0.2, random_state=2)
```

This produces 6,818 training rows and 1,705 holdout rows. The model is `xgboost.XGBRegressor` instantiated with its library defaults.

## Results

The saved notebook output reports the following coefficient of determination (R²) values:

| Partition | R² |
| --- | ---: |
| Training set | 0.8762 |
| Holdout set | 0.5017 |

These results are a baseline from one fixed split, not a production-quality performance estimate. The gap between training and holdout R² indicates that additional validation and tuning are warranted.

## Reproducing the workflow in Python

The central training and evaluation steps are equivalent to:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

X = big_mart_data.drop(columns="Item_Outlet_Sales")
y = big_mart_data["Item_Outlet_Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

model = XGBRegressor()
model.fit(X_train, y_train)

print("Train R²:", r2_score(y_train, model.predict(X_train)))
print("Holdout R²:", r2_score(y_test, model.predict(X_test)))
```

`big_mart_data` in the snippet assumes the imputation, fat-content cleanup, and encoding steps documented above have already been applied.

## Limitations and next steps

- **Prediction export:** add preprocessing for `sample_data/Test.csv`, call `model.predict(...)`, and write predictions in the required submission format.
- **Reusable preprocessing:** fit categorical encoders on the training partition and reuse them for validation and inference rather than encoding the entire dataset before the split.
- **Validation:** use cross-validation and report multiple metrics, such as RMSE and MAE, in addition to R².
- **Tuning:** tune XGBoost hyperparameters (for example, number of estimators, learning rate, depth, and subsampling) with a validation strategy.
- **Feature engineering:** consider outlet age, item-category signals, and treatment of zero visibility values.
- **Reproducibility:** pin dependency versions in a `requirements.txt` or equivalent environment file and set all applicable random seeds.

## License

No license file is currently included. Add an explicit license before redistributing or reusing the project beyond its intended context.
