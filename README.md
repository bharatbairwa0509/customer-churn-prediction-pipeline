\# Customer Churn Prediction Pipeline



An end-to-end \*\*Data Engineering + Machine Learning pipeline\*\* for predicting customer churn using \*\*PySpark and Spark MLlib\*\*.



The project takes raw customer data through validation, cleaning, feature engineering, model training, evaluation, pipeline serialization, and batch prediction on new customers.



\---



\## 📌 Project Overview



Customer churn prediction helps businesses identify customers who are likely to discontinue their services.



This project demonstrates how to build a production-style machine learning workflow using \*\*PySpark\*\*, rather than performing all preprocessing and modeling inside a single notebook.



The pipeline includes:



\* Data validation

\* Data cleaning

\* Feature engineering

\* Train/test splitting

\* Categorical encoding

\* Feature vectorization

\* Random Forest classification

\* Model evaluation

\* Feature importance analysis

\* Complete Spark ML pipeline saving

\* Batch prediction on new customer data

\* Churn probability generation



\---



\## 🏗️ Project Architecture



```text

&#x20;                   Customer Churn Prediction Pipeline



&#x20;                             Raw CSV

&#x20;                               │

&#x20;                               ▼

&#x20;                      Data Validation

&#x20;                               │

&#x20;                               ▼

&#x20;                        Data Cleaning

&#x20;                               │

&#x20;                               ▼

&#x20;                     Feature Engineering

&#x20;                               │

&#x20;                               ▼

&#x20;                        Train / Test Split

&#x20;                               │

&#x20;                               ▼

&#x20;                   ┌───────────────────────┐

&#x20;                   │     Spark ML Pipeline │

&#x20;                   │                       │

&#x20;                   │  StringIndexer        │

&#x20;                   │        ↓              │

&#x20;                   │  OneHotEncoder        │

&#x20;                   │        ↓              │

&#x20;                   │  VectorAssembler      │

&#x20;                   │        ↓              │

&#x20;                   │  Random Forest        │

&#x20;                   └───────────────────────┘

&#x20;                               │

&#x20;                               ▼

&#x20;                      Saved Pipeline Model

&#x20;                               │

&#x20;                               ▼

&#x20;                      New Customer CSV

&#x20;                               │

&#x20;                               ▼

&#x20;                        Batch Prediction

&#x20;                               │

&#x20;                               ▼

&#x20;                Prediction + Churn Probability

```



\---



\## 🎯 Business Problem



The objective is to predict whether a customer is likely to churn based on information such as:



\* Customer tenure

\* Monthly charges

\* Total charges

\* Contract type

\* Internet service

\* Technical support

\* Online security

\* Payment method

\* Demographic information

\* Other subscribed services



The model produces:



1\. A churn prediction: `Yes` or `No`

2\. A probability representing the model's estimated likelihood of churn



\---



\## 📊 Dataset



The project uses the \*\*Telco Customer Churn dataset\*\*.



Original dataset file:



```text

WA\_Fn-UseC\_-Telco-Customer-Churn.csv

```



The dataset contains customer information and the target variable:



```text

Churn

```



Target values:



```text

Yes

No

```



\---



\## 🧹 Data Validation



The project includes a dedicated validation script:



```text

src/data/validation.py

```



The validation process checks:



\* Required columns

\* NULL values

\* Duplicate customer IDs

\* Invalid `SeniorCitizen` values

\* Invalid `tenure` values

\* Invalid `MonthlyCharges` values

\* Invalid `TotalCharges` values

\* Invalid `Churn` values



Example validation result:



```text

Dataset loaded successfully!

Rows: 7043

Columns: 21



All required columns are present.



Duplicate customer IDs: 0

Invalid SeniorCitizen values: 0

Invalid tenure values: 0

Invalid MonthlyCharges values: 0

Invalid TotalCharges values: 0

Invalid Churn values: 0



DATA VALIDATION PASSED

```



\---



\## 🧹 Data Cleaning



Data cleaning is implemented in:



```text

src/data/cleaning.py

```



The cleaning process includes:



\* Reading the raw CSV using PySpark

\* Removing duplicate records

\* Trimming `TotalCharges`

\* Converting blank `TotalCharges` values to NULL

\* Casting `TotalCharges` to `double`

\* Checking invalid values

\* Saving the cleaned dataset



Output:



```text

data/processed/cleaned\_churn/

```



\---



\## ⚙️ Feature Engineering



Feature engineering is implemented in:



```text

src/features/feature\_engineering.py

```



The project creates additional business-oriented features.



\### Average Monthly Charge



```text

AverageMonthlyCharge =

TotalCharges / tenure

```



For customers with zero tenure, the value is handled safely.



\### IsNewCustomer



Customers with tenure of 6 months or less:



```text

IsNewCustomer = 1

```



Otherwise:



```text

IsNewCustomer = 0

```



\### IsMonthToMonth



```text

Month-to-month → 1

Other contracts → 0

```



\### HasTechSupport



```text

TechSupport = Yes → 1

Otherwise → 0

```



\### HasOnlineSecurity



```text

OnlineSecurity = Yes → 1

Otherwise → 0

```



\### HasStreaming



If either StreamingTV or StreamingMovies is `Yes`:



```text

HasStreaming = 1

```



Otherwise:



```text

HasStreaming = 0

```



Engineered data is saved to:



```text

data/processed/engineered\_churn/

```



\---



\## 🤖 Machine Learning Pipeline



The machine learning workflow is implemented in:



```text

src/models/train.py

```



The data is split into:



```text

80% Training

20% Testing

```



\### Preprocessing



Categorical features are processed using:



```text

StringIndexer

&#x20;       ↓

OneHotEncoder

&#x20;       ↓

VectorAssembler

```



The target variable is converted into a numerical label:



```text

No  → 0

Yes → 1

```



\### Model



The final model is:



```text

Random Forest Classifier

```



Configuration:



```text

numTrees = 50

maxDepth = 8

seed = 42

```



\---



\## 🔒 Leakage-Safe Pipeline



A key design decision in this project is that preprocessing is fitted \*\*only on the training data\*\*.



Instead of separately encoding the training and prediction datasets, the complete preprocessing and model workflow is combined into one Spark ML pipeline:



```text

StringIndexer

&#x20;    ↓

OneHotEncoder

&#x20;    ↓

VectorAssembler

&#x20;    ↓

RandomForestClassifier

```



The fitted pipeline is then saved as a single model.



This allows the same transformations learned during training to be reused during prediction.



Saved model:



```text

models/churn\_pipeline/

```



\---



\## 📈 Model Evaluation



Final leakage-safe Random Forest results:



| Metric            |     Result |

| ----------------- | ---------: |

| Accuracy          | \*\*80.74%\*\* |

| Weighted F1 Score | \*\*79.30%\*\* |

| Churn Precision   | \*\*70.82%\*\* |

| Churn Recall      | \*\*46.35%\*\* |



\### Confusion Matrix



```text

&#x20;                   Predicted

&#x20;                   No      Yes



Actual No           921      68

Actual Yes          191     165

```



Where:



```text

True Negative  = 921

False Positive = 68

False Negative = 191

True Positive  = 165

```



The project focuses specifically on \*\*churn-class precision and recall\*\*, rather than relying only on overall accuracy.



\---



\## 🔍 Feature Importance



The Random Forest model was also analyzed to understand which features contributed most to the model's predictions.



Top features included:



| Rank | Feature                       | Importance |

| ---: | ----------------------------- | ---------: |

|    1 | Contract — Month-to-month     |   0.124764 |

|    2 | tenure                        |   0.121163 |

|    3 | IsMonthToMonth                |   0.088445 |

|    4 | OnlineSecurity — No           |   0.069220 |

|    5 | TotalCharges                  |   0.069067 |

|    6 | InternetService — Fiber optic |   0.048507 |

|    7 | IsNewCustomer                 |   0.046600 |

|    8 | MonthlyCharges                |   0.046494 |

|    9 | TechSupport — No              |   0.045858 |

|   10 | AverageMonthlyCharge          |   0.044630 |



These importances describe the features used by this particular trained Random Forest and should not be interpreted as causal effects.



\---



\## 🔮 Batch Prediction



Prediction is implemented in:



```text

src/models/predict.py

```



The prediction workflow accepts a new customer CSV:



```text

New Customer CSV

&#x20;      ↓

Saved Spark Pipeline

&#x20;      ↓

Feature Engineering

&#x20;      ↓

Prediction

&#x20;      ↓

Churn Probability

```



Example input:



```csv

customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges

NEW001,Male,0,No,No,2,Yes,No,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,70.50,141.00

```



Example prediction output:



```text

+----------+----------------+------------------+

|customerID|churn\_prediction|churn\_probability |

+----------+----------------+------------------+

|NEW001    |No              |0.4473143838      |

+----------+----------------+------------------+

```



For the test customers used in this project:



```text

NEW001 → No  → 44.73%

NEW002 → Yes → 53.90%

NEW003 → No  → 2.70%

```



The `churn\_probability` represents the model's estimated probability for the `Yes` churn class.



Predictions are saved to:



```text

predictions/

```



\---



\## 📁 Project Structure



```text

Customer Churn Prediction Pipeline/

│

├── data/

│   ├── raw/

│   │   └── WA\_Fn-UseC\_-Telco-Customer-Churn.csv

│   │

│   └── processed/

│       ├── cleaned\_churn/

│       └── engineered\_churn/

│

├── notebooks/

│   ├── 01\_EDA.ipynb

│   └── 02\_Model\_Experiments.ipynb

│

├── src/

│   ├── data/

│   │   ├── cleaning.py

│   │   └── validation.py

│   │

│   ├── features/

│   │   └── feature\_engineering.py

│   │

│   └── models/

│       ├── train.py

│       └── predict.py

│

├── models/

│   └── churn\_pipeline/

│

├── reports/

│   └── model\_metrics.json

│

├── predictions/

│

├── README.md

├── requirements.txt

└── .gitignore

```



\---



\## 🛠️ Technologies Used



\### Programming



\* Python



\### Data Engineering



\* PySpark

\* Spark SQL

\* PySpark DataFrames



\### Machine Learning



\* Spark MLlib

\* Random Forest

\* StringIndexer

\* OneHotEncoder

\* VectorAssembler



\### Data Analysis



\* Pandas

\* Matplotlib

\* YData Profiling



\### Development Environment



\* Google Colab

\* Google Drive

\* GitHub



\---



\## 🚀 How to Run



\### 1. Install dependencies



```bash

pip install -r requirements.txt

```



\### 2. Validate the data



```bash

python src/data/validation.py

```



\### 3. Clean the data



```bash

python src/data/cleaning.py

```



\### 4. Perform feature engineering



```bash

python src/features/feature\_engineering.py

```



\### 5. Train the model



```bash

python src/models/train.py

```



The trained Spark pipeline will be saved to:



```text

models/churn\_pipeline/

```



\### 6. Generate predictions



```bash

python src/models/predict.py test\_customers.csv

```



Predictions will be generated and saved to:



```text

predictions/

```



\---



\## 📋 Output



The prediction pipeline produces:



```text

customerID

churn\_prediction

churn\_probability

```



Example:



```text

NEW001    No     0.4473

NEW002    Yes    0.5390

NEW003    No     0.0270

```



\---



\## 💡 Key Project Learnings



This project demonstrates practical experience with:



\* Building an end-to-end ML pipeline

\* PySpark DataFrame processing

\* Data validation

\* Data cleaning

\* Feature engineering

\* Categorical encoding

\* Train/test splitting

\* Avoiding preprocessing leakage

\* Spark ML pipelines

\* Random Forest classification

\* Model evaluation

\* Feature importance analysis

\* Model serialization

\* Batch prediction

\* Generating prediction probabilities



\---



\## 🔄 End-to-End Execution



The complete workflow is:



```bash

python src/data/validation.py

python src/data/cleaning.py

python src/features/feature\_engineering.py

python src/models/train.py

python src/models/predict.py test\_customers.csv

```



\---



\## 📌 Future Improvements



Possible future improvements include:



\* Hyperparameter tuning

\* Cross-validation

\* Class imbalance handling

\* Experiment tracking

\* Model comparison

\* Automated data validation

\* Model monitoring

\* REST API for real-time predictions

\* Dockerization

\* Cloud deployment

\* Automated CI/CD pipeline



\---



\## 👨‍💻 Author



\*\*Bharat Bairwa\*\*



BCA — Artificial Intelligence \& Data Science



Interested in:



\* Machine Learning

\* Data Engineering

\* Python

\* PySpark

\* AI/ML Applications

\* Software Engineering



\---



\## ⭐ Project Summary



This project demonstrates an end-to-end \*\*Customer Churn Prediction Pipeline using PySpark and Spark MLlib\*\*, covering the complete workflow from raw data validation and feature engineering to model training, model serialization, and batch prediction.



The main objective is to demonstrate practical \*\*Data Engineering + Machine Learning pipeline development\*\* rather than only building a model inside a notebook.



