import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib

### Read data

df = pd.read_csv('Clean_Dataset.csv')
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()


### Data Cleaning

df = df.drop(columns='Unnamed: 0')
df.head()
count = df.airline.value_counts()
percentage = df.airline.value_counts(normalize=True)*100
freq_table = pd.DataFrame({'Frequency':count,'Percent':percentage})

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8')
ax = sns.barplot(x=freq_table.index, y="Frequency", data=freq_table, palette="viridis")

for i, (freq, perc) in enumerate(zip(freq_table['Frequency'], freq_table['Percent'])):
    ax.text(i, freq + 0.5, f'{freq} ({perc:.1f}%)', ha='center', fontsize=10, color='black')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.xlabel("Airline", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Airline", fontsize=16)
plt.tight_layout()
plt.show()

counts = df['source_city'].value_counts()
sns.barplot(y=counts.index,x=counts,palette='viridis')
plt.title('Source City',fontsize=18,fontweight='600')
plt.xlabel('Count',fontsize=18)
plt.ylabel('City',fontsize=18)
plt.show()


counts = df['departure_time'].value_counts()
sns.barplot(y=counts.index,x=counts,palette='viridis')
plt.title('Departure Time',fontsize=18,fontweight='600')
plt.xlabel('Count',fontsize=18)
plt.ylabel('Time',fontsize=18)
plt.show()


counts = df['stops'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=['One Stop','Zero Stops','Two Or More'], autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))  
plt.title("Stops", fontsize=18)
plt.show()


counts = df['arrival_time'].value_counts()
sns.barplot(y=counts.index,x=counts,palette='viridis')
plt.title('Arrival Time',fontsize=18,fontweight='600')
plt.xlabel('Count',fontsize=18)
plt.ylabel('Time',fontsize=18)
plt.show()


counts = df['destination_city'].value_counts()
sns.barplot(y=counts.index,x=counts,palette='viridis')
plt.title('Destination',fontsize=18,fontweight='600')
plt.xlabel('Count',fontsize=18)
plt.ylabel('City',fontsize=18)
plt.show()


counts = df['class'].value_counts()
sns.barplot(x=counts.index,y=counts,palette='viridis')
plt.title('Destination',fontsize=18,fontweight='600')
plt.ylabel('Count',fontsize=18)
plt.xlabel('City',fontsize=18)
plt.show()


df.duration.describe()
plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
sns.histplot(df['duration'], bins=30, kde=True)
plt.title('Trip Duration')
plt.xlabel('Hours')
plt.ylabel('Count')
plt.show()


df['days_left'].describe()
plt.figure(figsize=(10, 6))
sns.kdeplot(df['days_left'], fill=True, color="skyblue", alpha=0.5)
plt.title("Days Left for the Trip")
plt.xlabel("Days")
plt.ylabel("Density")
plt.show()


df.price.describe()
sns.boxplot(x=df['price'])
plt.show()

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(outliers)

outliers.duration.describe()
outliers['class'].value_counts()
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
df.info()




#### Data Processing

df = df.drop(columns='flight')
categorical_cols = df.iloc[:,:7]
categorical_cols.head()

label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation')
plt.show()



#### Select Features & Target
X = df.drop(columns='price')
y = df['price'] 

### Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=42)

### ML Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    #'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    #'SVR': SVR(kernel='rbf'),
    #'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    #'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    #'XGBoost Regressor': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100),
    #'CatBoost Regressor': catboost.CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0),
    #'LightGBM Regressor': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
}

results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)  
    test_score = model.score(X_test, y_test)     
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results.append({
        'Model': model_name,
        'Train R² Score': train_score,
        'Test R² Score': test_score,
        'Mean Squared Error': mse

    })

results_df = pd.DataFrame(results)
print(results_df)


# Train and Save Models 
for model_name, model in models.items(): 
    model.fit(X_train, y_train) 
    joblib.dump(model, f'{model_name.lower().replace(" ", "_")}_model.joblib')
    
print("All models trained and saved successfully.")