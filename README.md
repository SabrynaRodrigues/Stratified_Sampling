
# 📊 Stratified Sampling with `train_test_split`

This notebook demonstrates how to use **stratified sampling** with the `train_test_split` function from Scikit-Learn. The goal is to split datasets in a way that preserves the proportion of classes across training and test sets.

---

## 📁 Dataset 1: Iris Dataset

We use the classic Iris dataset to demonstrate stratification.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.csv')
iris['class'].value_counts()
```

### 🎯 Stratified Split

We split the dataset so that the proportions of each flower class are preserved:

```python
X, _, y, _ = train_test_split(
    iris.iloc[:, :4],       # columns 0 to 3 as independent variables
    iris.iloc[:, 4],        # column 4 as the dependent variable (class)
    test_size = 0.5,        # 50% split
    stratify = iris.iloc[:, 4]  # use class column to stratify
)

y.value_counts()
```

---

## 📁 Dataset 2: Infert Dataset

We use a dataset named `infert.csv`, where we stratify based on the `education` column.

```python
infert = pd.read_csv('infert.csv')
infert
```

### 🔍 Count Values in 'education' Column

```python
infert['education'].value_counts()
```

### 🎯 Stratified Split

We now stratify the data based on the `education` column while keeping the other columns as features:

```python
X1, _, y1, _ = train_test_split(
    infert.iloc[:, 2:9],     # columns 2 to 8 as independent variables
    infert.iloc[:, 1],       # column 1 is the education level
    test_size = 0.6,         # 60% split
    stratify = infert.iloc[:, 1]  # use education column to stratify
)

y1.value_counts()
```

---

