import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

# Changing file path to be the same as the dataset
project_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_folder)

# Importing dataset
data = pd.read_csv("dataset/diabetes.csv")

# Cleaning the dataset from zeros and spaces
data.replace({0: pd.NA, " ": pd.NA}, inplace=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    
    if col == "Outcome":
        data[col] = data[col].fillna(0) 
    else:
        mean_value = data[col].mean(skipna=True)
        data[col] = data[col].fillna(mean_value)

    data[col] = data[col].round(3)

# Calculating statistic values
means = {}
for col in data.columns:
    col_mean = np.mean(data[col])
    means[col] = col_mean

modes = {}
for col in data.columns:  
    col_mode = data[col].mode(dropna=True)
    modes[col] = col_mode[0]  

medians = {}
for col in data.columns:
    col_median = data[col].median(skipna=True)
    medians[col] = col_median

stds = {}
for col in data.columns:
    col_std = data[col].std(skipna=True)
    stds[col] = col_std

# Printing the results of the statistic calculations
os.system("cls")

print("-The mean of it's column-")
for key, value in means.items():
    print(f"{key}: {value: .3f}")
os.system("pause")
os.system("cls")

print("-The mode of it's column-")
for key, value in modes.items():
    print(f"{key}: {value: .3f}")
os.system("pause")
os.system("cls")

print("-The median of it's column-")
for key, value in medians.items():
    print(f"{key}: {value: .3f}")   
os.system("pause")
os.system("cls")

print("-The std of it's column-")
for key, value in stds.items():
    print(f"{key}: {value: .3f}")    
os.system("pause")
os.system("cls")

# Visualisation of the dataset
for col in data.columns:
    sns.histplot(data[col], kde=True)
    plt.show()

corr_matrix = data.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()