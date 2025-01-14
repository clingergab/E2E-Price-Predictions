import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Data/Real_Estate.csv")

print(data.head())
print(data.info())
print(data.describe())


# Create histograms for the numerical columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Histograms of Real Estate Data', fontsize=16)

cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
        'Latitude', 'Longitude', 'House price of unit area']

for i, col in enumerate(cols):
    sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# Scatter plots to observe the relationship with house price
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

# Scatter plot for each variable against the house price
sns.scatterplot(data=data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# Heatmap for correlations
plt.figure(figsize=(10, 10))
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('correlation heatmap')
# plt.show()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(data['Latitude'], data['Longitude'], data['House price of unit area'], 
                      c=data['House price of unit area'], cmap='viridis', alpha=0.7)

# Add labels
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('House Price of Unit Area')
plt.colorbar(scatter, label='House Price')
plt.title('3D Scatter Plot: Latitude, Longitude, and House Price')
# plt.show()

plt.figure(figsize=(20, 12))
sns.boxplot(data=data)
plt.xticks(rotation=90)
# plt.show()

# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'House age']
target = 'House price of unit area'

X = data[features]
y = data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions using the linear regression model
y_pred_lr = model.predict(X_test)

# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
# plt.show()

print(mean_squared_error(y_test, y_pred_lr))
print(r2_score(y_test, y_pred_lr))

plt.figure(figsize=(20, 12))
sns.boxplot(data=X)
plt.xticks(rotation=90)
# plt.show()

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test) 

# Model initialization
model2 = LinearRegression()

# Training the model
model2.fit(X_train_scale, y_train)

# Making predictions using the linear regression model
y_pred_lr2 = model2.predict(X_test_scale)

# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr2, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
# plt.show()

print(mean_squared_error(y_test, y_pred_lr2))
print(r2_score(y_test, y_pred_lr2))


