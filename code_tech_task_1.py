
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# View the first few rows of the dataset
print(iris.head())

#Basic Info
# Check the data types and null values
print(iris.info())

# Statistical summary of the dataset
print(iris.describe())

# Check for any missing values
print(iris.isnull().sum())

#Unvariate Analysis

# Set up the matplotlib figure
plt.figure(figsize=(12, 6))

# Plot histograms for each numeric column
iris.hist(edgecolor='black', linewidth=1.2, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Kernel Density Estimation (KDE) plots for sepal_length
sns.kdeplot(iris['sepal_length'], shade=True)
plt.title('KDE Plot of Sepal Length')
plt.show()

#Multi Variate Analysis
# Scatter plot between Sepal Length and Sepal Width
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Pair plot to visualize relationships between features
sns.pairplot(iris, hue='species')
plt.show()

#Correlation Matrix and Heat Map
# Correlation matrix
corr_matrix = iris.drop('species', axis=1).corr() # Drop the non-numeric column 'species'
print(corr_matrix)

# Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplot for Sepal Length by Species
plt.figure(figsize=(6, 4))
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

# Violin plot for Petal Length by Species
plt.figure(figsize=(6, 4))
sns.violinplot(x='species', y='petal_length', data=iris)
plt.title('Violin Plot of Petal Length by Species')
plt.show()
