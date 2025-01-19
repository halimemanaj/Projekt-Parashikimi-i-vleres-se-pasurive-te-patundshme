# Projekt-Parashikimi-i-vleres-se-pasurive-te-patundshme
Parashikimi i Vleres se pasurive te patundshme

# Instalimi i Librarive
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

# Percaktimi i path te te dhenave
train_file_path = '/content/drive/MyDrive/train.csv'
test_file_path = '/content/drive/MyDrive/test.csv'

# Ngarkimi i datasetit
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Shembull Vizualizimi
sns.pairplot(df_train)
plt.show()
print

# Printimi i rreshtave të parë për të dy datasetet
print(df_train.head())
print(df_test.head())

# Ruajtja e ID-ve të të dhënave të trajnimit dhe testit për përdorim më vonë
train_ID = df_train['Id']
test_ID = df_test['Id']
df_train.drop("Id", axis=1, inplace=True)  # Heqja e kolonës 'Id' nga dataseti i trajnimit
df_test.drop("Id", axis=1, inplace=True)   # Heqja e kolonës 'Id' nga dataseti i testit

# Regjistrimi i madhësive të të dhënave të trajnimit dhe testit
ntrain = df_train.shape[0]
ntest = df_test.shape[0]

# Nxjerrja e variablës target "SalePrice" nga dataseti i trajnimit
y_train = df_train["SalePrice"]
df_train.drop("SalePrice", axis=1, inplace=True)  # Heqja e kolonës "SalePrice" nga dataseti i trajnimit
# Bashkimi i të dhënave të trajnimit dhe testit për përpunim të përbashkët
df_total = pd.concat((df_train, df_test)).reset_index(drop=True)
print(df_total.info())

# Llogaritja e të dhënave të munguar
missing_data = df_total.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_data / len(df_total)) * 100

# Bashkimi i të dhënave të munguar
missing_df = pd.DataFrame({'Total': missing_data, 'Percentage': missing_percentage})

# Filtrohet për të treguar vetëm ato që kanë më shumë se 0 mungesa
missing = missing_df[missing_df['Total'] > 0]
print(missing)

# Krijimi i diagramit
plt.figure(figsize=(10, 6))
sns.barplot(x=missing.index, y=missing['Percentage'], color='blue')

# Titujt dhe etiketat për diagramin
plt.title('Përqindja e të Dhënave të Munguar për Çdo Karakteristikë', fontsize=16)
plt.xlabel('Karakteristikat', fontsize=12)
plt.ylabel('Përqindja e Mungesës (%)', fontsize=12)
plt.xticks(rotation=90)  # Rrotullimi i etiketave në boshtin X
plt.tight_layout()

# Shfaqja e diagramit
plt.show()

def fill_missing_values(df, col_dict):
    for col, value in col_dict.items():
        df[col] = df[col].fillna(value)  # Plotësimi i mungesave nëpërmjet vlerave të dhëna
    return df

# Definimi i vlerave për plotësimin e mungesave
fill_dict = {
    "PoolQC": "None", "MiscFeature": "None", "Alley": "None", "Fence": "None",
    "FireplaceQu": "None", "MasVnrType": "None", "MSZoning": df_total['MSZoning'].mode()[0],
    "Functional": df_total['Functional'].mode()[0], "Electrical": df_total['Electrical'].mode()[0],
    "KitchenQual": df_total['KitchenQual'].mode()[0], "Exterior1st": df_total['Exterior1st'].mode()[0],
    "Exterior2nd": df_total['Exterior2nd'].mode()[0], "SaleType": df_total['SaleType'].mode()[0],
    "MSSubClass": "None"
}

# Aplikimi i plotësimit të mungesave
df_total = fill_missing_values(df_total, fill_dict)

# Plotësimi i 'LotFrontage' duke përdorur median e lagjes
df_total['LotFrontage'] = df_total.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Plotësimi i mungesave për grupet e kolonave specifike
cols_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
df_total[cols_none] = df_total[cols_none].apply(lambda x: x.fillna('None'))
cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
df_total[cols_zero] = df_total[cols_zero].fillna(0)

# Heqja e kolonave që nuk janë të nevojshme
df_total.drop(['Utilities'], axis=1, inplace=True)

# Sigurimi që të gjitha mungesat janë adresuar
df_total['Functional'] = df_total['Functional'].fillna(df_total['Functional'].mode()[0])
df_total['Electrical'] = df_total['Electrical'].fillna(df_total['Electrical'].mode()[0])
df_total['KitchenQual'] = df_total['KitchenQual'].fillna(df_total['KitchenQual'].mode()[0])
df_total['Exterior1st'] = df_total['Exterior1st'].fillna(df_total['Exterior1st'].mode()[0])
df_total['Exterior2nd'] = df_total['Exterior2nd'].fillna(df_total['Exterior2nd'].mode()[0])
df_total['SaleType'] = df_total['SaleType'].fillna(df_total['SaleType'].mode()[0])

# Krijimi i karakteristikave të reja bazuar në kolonat ekzistuese
df_total['TotalLivingArea'] = df_total['TotalBsmtSF'] + df_total['1stFlrSF'] + df_total['2ndFlrSF']
df_total['HouseAge'] = df_total['YrSold'] - df_total['YearBuilt']
df_total['RemodAge'] = df_total['YrSold'] - df_total['YearRemodAdd']
df_total['GardenArea'] = df_total['LotArea'] - df_total['TotalBsmtSF'] - df_total['GrLivArea']

# Konvertimi i kolonave numerike me kuptime specifike në vargje
df_total["MSSubClass"] = df_total["MSSubClass"].astype(str)
df_total['MoSold'] = df_total['MoSold'].astype(str)
print(df_total.info())

# Analiza e shpërndarjes së variablave numerikë
numerical_features = df_total.select_dtypes(include=["int64", "float64"]).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_total[feature], bins=50, kde=True, color="green")
    plt.title(f"Shpërndarja e {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frekuenca")
    plt.show()

# Shpërndarja e frekuencës së variablave kategorikë
categorical_features = df_total.select_dtypes(include=["object"]).columns
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_total, x=feature, color="red")
    plt.title(f"Grafiku i Numrit për {feature}")
    plt.xlabel(feature)
    plt.ylabel("Numri")
    plt.xticks(rotation=45)
    plt.show()

# Marrëdhënia mes variablës target dhe variablave numerikë (grafiku i shpërndarjes)
for feature in numerical_features:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df_total[feature][:ntrain], y=y_train, color="purple")
    plt.title(f"{feature} vs SalePrice")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.show()

# Marrëdhënia mes variablës target dhe variablave kategorikë (grafiku kutie)
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_total[feature][:ntrain], y=y_train, color="blue")
    plt.title(f"{feature} vs SalePrice")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.xticks(rotation=45)
    plt.show()

import numpy as np

# Selektoni kolonat numerike nga dataframe
df_numeric = df_total.select_dtypes(include=[np.number])

# Krijimi i matrisë së korrelacionit
corr = df_numeric.corr()

# Maskimi i trekëndëshit të sipërm për të injoruar korrelacionet e vetvetes (vlerat diagonale)
mask = np.triu(np.ones(corr.shape), k=1)
corr_masked = corr.where(mask == 0)

# Heqja e korrelacioneve të vetvetes (diagonala, që janë 1.0)
corr_masked = corr_masked[corr_masked != 1.0]

# Marrja e 10 korrelacioneve më të larta
top_10_corr = (
    corr_masked.unstack()   # Konvertimi i matrices në seri
    .dropna()               # Heqja e vlerave NaN
    .abs()                  # Marrja e vlerave absolute të korrelacioneve
    .sort_values(ascending=False)  # Renditja nga korrelacionet më të larta
    .head(10)               # Marrja e 10 korrelacioneve më të larta
)

# Konvertimi i korrelacioneve më të larta në një DataFrame për vizualizim më të mirë
top_10_corr_df = top_10_corr.reset_index()
top_10_corr_df.columns = ['Karakteristika 1', 'Karakteristika 2', 'Korrelacioni']

# Krijimi i një tabele pivot për vizualizimin e matrisë 2D të korrelacioneve
corr_matrix = top_10_corr_df.pivot_table(index='Karakteristika 1', columns='Karakteristika 2', values='Korrelacioni')

# Vizualizimi i përmirësuar i hartës ngrohtë
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True, fmt=".4f", linewidths=0.5, linecolor='gray')
plt.title("10 Korrelacionet Më të Larta")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder, RobustScaler
# Kodimi i variablave kategorikë dhe normalizimi i variablave numerikë
label_encoder = LabelEncoder()
categorical_cols = df_total.select_dtypes(include=['object', 'category']).columns

# Përdorimi i LabelEncoder për të koduar kolonat kategorike
for col in categorical_cols:
    df_total[col] = label_encoder.fit_transform(df_total[col])

# Selektoni kolonat numerike
numerical_cols = df_total.select_dtypes(include=[np.number]).columns
scaler_robust = RobustScaler()
df_total[numerical_cols] = scaler_robust.fit_transform(df_total[numerical_cols])

# Pjesëtimi i të dhënave në trajnimin dhe testimin e dataset-it
df_train = df_total[:ntrain]
df_test = df_total[ntrain:]

# Definimi i të dhënave të trajnimit dhe testimit
X_train = df_train
X_test = df_test

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Definimi i modeleve dhe hiperparametrave
models = {
    "DecisionTreeRegressor": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'rbf'],
            "gamma": ['scale', 'auto']
        }
    },
    "KNeighborsRegressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 10],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }
}

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Kryerja e GridSearchCV për çdo model
best_models = {}
for model_name, model_info in models.items():
    # Përdorimi i GridSearchCV për optimizimin e hiperparametrave të modeleve
    grid_search = GridSearchCV(estimator=model_info["model"], param_grid=model_info["params"], cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Bërja e parashikimeve me modelet më të mira
predictions = {}
for model_name, model in best_models.items():
    predictions[model_name] = model.predict(X_test)

# Vlerësimi i secilit model duke përdorur MSE, MAE, RMSE, dhe R2
metrics = {}
for model_name, y_pred in predictions.items():
    mse = mean_squared_error(y_train, best_models[model_name].predict(X_train))
    mae = mean_absolute_error(y_train, best_models[model_name].predict(X_train))
    r2 = r2_score(y_train, best_models[model_name].predict(X_train))
    rmse = np.sqrt(mse)

    metrics[model_name] = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "RMSE": rmse
    }

# Shfaqja e rezultateve për secilin model
for model_name, metric in metrics.items():
    print(f"{model_name}:")
    print(f"  MSE: {metric['MSE']}")
    print(f"  MAE: {metric['MAE']}")
    print(f"  R2: {metric['R2']}")
    print(f"  RMSE: {metric['RMSE']}")
    print()

# Gjetja e modelit më të mirë në bazë të scorës R2
best_model_name = max(metrics, key=lambda x: metrics[x]["R2"])
best_model = best_models[best_model_name]

# Gjenerimi i parashikimeve për setin e testimit duke përdorur modelin më të mirë
ensemble = best_model.predict(X_test)

# Krahasimi i performancës së modeleve nëpërmjet grafikëve (grafikë shiritash)
metrics_df = pd.DataFrame(metrics).T

# Krijimi i subplots për çdo metrikë
fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Krijimi i 3 subplots

# Grafik për MSE
metrics_df['MSE'].plot(kind='bar', ax=ax[0], color='blue', title="MSE", legend=False)
ax[0].set_ylabel("Vlera e Gabimit")
ax[0].set_xlabel("Modelet")

# Grafik për MAE
metrics_df['MAE'].plot(kind='bar', ax=ax[1], color='orange', title="MAE", legend=False)
ax[1].set_ylabel("Vlera e Gabimit")
ax[1].set_xlabel("Modelet")

# Grafik për RMSE
metrics_df['RMSE'].plot(kind='bar', ax=ax[2], color='green', title="RMSE", legend=False)
ax[2].set_ylabel("Vlera e Gabimit")
ax[2].set_xlabel("Modelet")

# Rregullimi i layout-it dhe shfaqja e grafikut
plt.tight_layout()
plt.show()

print(metrics_df)

# Grafik i krahasimit mes parashikimeve dhe vlerave reale për modelin më të mirë
best_train_preds = best_model.predict(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(y_train, best_train_preds, alpha=0.7, edgecolors='k', color='purple')
plt.title(f"{best_model_name} Model - Vlerat Reale vs Parashikimet")
plt.xlabel("Vlerat Reale")
plt.ylabel("Parashikimet")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # Linea e kuqe për barazinë mes vlerave reale dhe atyre të parashikuara
plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance

# Funksioni për vizualizimin e rëndësisë së karakteristikave më të rëndësishme
def plot_feature_importance(model, X_train, model_name, top_n=10):
    if model_name in ['SVR', 'KNeighborsRegressor']:
        # Për modelet pa atributin feature_importances_, përdorim rëndësinë e permutimit
        results = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        importances = results.importances_mean
    else:
        # Për modelet me atributin feature_importances_
        importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices_top_n = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Rëndësia e Karakteristikave - {model_name}")
    plt.barh(range(top_n), importances[indices_top_n], align="center")
    plt.yticks(range(top_n), [X_train.columns[i] for i in indices_top_n])
    plt.xlabel("Rëndësia Relative")
    plt.tight_layout()
    plt.show()

# Vizualizimi i rëndësisë së karakteristikave për çdo model
for model_name, model in best_models.items():
    plot_feature_importance(model, X_train, model_name)

# Ruajtja e parashikimeve në një skedar CSV
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = ensemble
submission.to_csv('submission.csv', index=False)

print(f"Parashikimet janë ruajtur në 'submission.csv' duke përdorur modelin më të mirë: {best_model_name}.")

