# Import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# Style
sns.set_palette("deep", 8)

# Load data
# ---------
train_data = pd.read_csv('train.csv')
train_data.drop("Id", axis = 1, inplace = True)
test_data = pd.read_csv('test.csv')
# Save test Id for submission
test_ID = test_data['Id']
test_data.drop("Id", axis = 1, inplace = True)

# Set train y
y_train = train_data['SalePrice']


# Concatenate both train and test into one complete df and remove target variable
full_data = pd.concat((train_data, test_data)).reset_index(drop=True)
full_data.drop(['SalePrice'], axis=1, inplace=True)


# Pre-processing
# --------------
# Fill NA data that means "NO" with None
for f in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Alley', 'FireplaceQu'):
    full_data[f] = full_data[f].fillna('None')

# Fix numerical categories that are set as numerical
full_data['MSSubClass'] = full_data['MSSubClass'].apply(str)
full_data['OverallCond'] = full_data['OverallCond'].astype(str)
full_data['YrSold'] = full_data['YrSold'].astype(str)
full_data['MoSold'] = full_data['MoSold'].astype(str)

# Get percentage of missing data
percent = (full_data.isnull().sum()/full_data.isnull().count()).sort_values(ascending=False)
print(percent[percent>0])

# Predict missing numerical values using most correlated features, encode categorical data, and predict missing categorical data
def fillMissingTop():
    # Fill missing data using Model
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Set paramaters
    DEFAULT_NUMBER_OF_TOP_CORRELATED_FEATURES = 5
    MINIMUM_NUMBER_OF_CORRELATED_FEATURES = 4
    DATA = full_data  # Don't forget to drop target y of the dataset

    # Number of features that have missing values
    remaining_numerical_features = 999
    remaining_categorical_features = 999

    # Number of top correlated features tracker
    number_of_top_correlated_features = DEFAULT_NUMBER_OF_TOP_CORRELATED_FEATURES

    # Predict Numerical variables
    # -----------------------------
    print("------------------------------------")
    print("Start predicting numerical variables")
    print("------------------------------------")
    while(remaining_numerical_features>0):
        # Get top missing features sorted that have more than 0 missing value
        features_missing = (DATA.isnull().sum()/DATA.isnull().count()).sort_values(ascending=False)
        features_missing = features_missing[features_missing>0].index

        # Divide missing features into categories
        numerical_features_missing = []
        for i, f in enumerate(full_data[features_missing].dtypes):
            if(f in ['int64','float64', 'float32','int32']):
                numerical_features_missing.append(features_missing[i])


        # Number of features missing
        remaining_numerical_features = len(numerical_features_missing)
        print(f"Numerical features remaining: {remaining_numerical_features}")

        # Compute the correlation matrix
        corr = DATA.dropna().corr()

        # Select feature with most missing values
        feature = numerical_features_missing[0]
        print(feature)

        # Select training data
        X_train_clean = DATA.dropna(axis=0, how='any')

        # Get top number 'number_of_top_correlated_features' of most correlated features
        top_correlated_features = list(corr[[feature]].abs().sort_values(by=feature, ascending=False)[1:number_of_top_correlated_features].index)

        # Select columns of the most correlated features
        X_train = X_train_clean[top_correlated_features]

        # Select target feature rows that aren't missing
        y_train = X_train_clean[[feature]]

        # Select test data (rows with missing 'feature')
        X_test_clean = DATA.loc[pd.isnull(DATA[[feature]]).any(axis=1)]

        # Select columns of the most correlated features and drop columns with NA
        X_test = X_test_clean[top_correlated_features].dropna(axis=1)

        # If after dropping there are less than MINIMUM_NUMBER_OF_CORRELATED_FEATURES features, increase the number_of_top_correlated_features to be considered
        if (X_test.shape[1]<MINIMUM_NUMBER_OF_CORRELATED_FEATURES):
            number_of_top_correlated_features += 1
            print(' -Pass')

        else:
            # Reset number_of_top_correlated_features (so the next missing feature starts from default)
            number_of_top_correlated_features = DEFAULT_NUMBER_OF_TOP_CORRELATED_FEATURES

            # Removing features not selected by X_test from X
            X_train = X_train[list(X_test.columns.values)]

            # Select target feature's missing rows
            yToFill = X_test_clean[[feature]]

            # Fit and predict using selected model
            regressor = RandomForestRegressor()
            regressor.fit(X_train, y_train.values.ravel())
            yPred = regressor.predict(X_test).flatten().tolist()

            # Replace NaN with predicted values
            yToFill = pd.DataFrame(yPred, columns=[feature])

            # Combine full dataset with predicted values and convert the feature to float (so it can be used later)
            #DATA[feature] = DATA[feature].combine_first(yToFill)
            #DATA[DATA.isnull()] = yToFill
            DATA.loc[DATA[feature].isnull(), feature] = yPred
            DATA[feature] = pd.to_numeric(DATA[feature], errors='coerce')

            # Decrease 1 from remaining number of features that have NA values
            remaining_numerical_features -= 1
            corr = DATA.dropna().corr()
            print(' -Done')

    # Encode categorical variables
    # ----------------------------
    print("----------------------------")
    print("Encode categorical variables")
    print("----------------------------")
    from sklearn import preprocessing

    # Select categorical variables
    #categorical_features = full_data.select_dtypes(include=['object'])
    categorical_features = []
    for i, f in enumerate(full_data.dtypes):
        if(f not in ['int64','float64', 'float32','int32']):
            categorical_features.append(full_data.columns[i])

    # Apply cat.codes
    full_data[categorical_features] = full_data[categorical_features].apply(lambda x: x.astype('category').cat.codes)

    # Convert -1 values back to NaN
    full_data[full_data[categorical_features] < 0] = np.NaN

    # Predict Categorical variables
    # -----------------------------
    print("--------------------------------------")
    print("Start predicting categorical variables")
    print("--------------------------------------")
    while(remaining_categorical_features>0):
        # Get top missing features sorted that have more than 0 missing value
        categorical_features_missing = (DATA[categorical_features].isnull().sum()/DATA[categorical_features].isnull().count()).sort_values(ascending=False)
        categorical_features_missing = categorical_features_missing[categorical_features_missing>0].index

        # Number of features missing
        remaining_categorical_features = len(categorical_features_missing)
        print(f"Categorical features remaining: {remaining_categorical_features}")

        # Select feature with most missing values
        feature = categorical_features_missing[0]
        print(feature)

        # Select target feature rows that aren't missing
        y_train = DATA[[feature]].dropna(axis=0, how='any')
        # Select training data, skip rows where the target feature isn't missing
        X_train = DATA.dropna(axis=1).iloc[list(y_train.index.values),:]

        # Select test data (rows with missing 'feature')
        X_test = DATA[pd.isnull(DATA[[feature]]).values][X_train.columns.values]

        # Fit and predict using selected model
        regressor = RandomForestClassifier()
        regressor.fit(X_train, y_train.values.ravel())
        yPred = regressor.predict(X_test).flatten().tolist()

        # Replace NaN with predicted values
        yToFill = pd.DataFrame(yPred, columns=[feature])

        # Combine full dataset with predicted values and convert the feature to float (so it can be used later)
        DATA.loc[DATA[feature].isnull(), feature] = yPred
        DATA[feature] = pd.to_numeric(DATA[feature], errors='coerce')

        # Decrease 1 from remaining number of features that have NA values
        remaining_categorical_features -= 1
        print(' -Done')

    return DATA

# Predict missing numerical values using most correlated features, encode categorical data, and predict missing categorical data
def fillMissingAll():
    # Fill missing data using Model
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Set paramaters
    DEFAULT_NUMBER_OF_TOP_CORRELATED_FEATURES = 5
    MINIMUM_NUMBER_OF_CORRELATED_FEATURES = 4
    DATA = full_data  # Don't forget to drop target y of the dataset

    # Number of features that have missing values
    remaining_numerical_features = 999
    remaining_categorical_features = 999

    # Number of top correlated features tracker
    number_of_top_correlated_features = DEFAULT_NUMBER_OF_TOP_CORRELATED_FEATURES

    # Predict Numerical variables
    # -----------------------------
    print("------------------------------------")
    print("Start predicting numerical variables")
    print("------------------------------------")
    while(remaining_numerical_features>0):
        # Get top missing features sorted that have more than 0 missing value
        features_missing = (DATA.isnull().sum()/DATA.isnull().count()).sort_values(ascending=False)
        features_missing = features_missing[features_missing>0].index

        # Divide missing features into categories
        numerical_features_missing = DATA[features_missing].dtypes[DATA.dtypes != "object"].index

        # Get complete list of numerical features
        numerical_features = DATA.dtypes[DATA.dtypes != "object"].index

        # Number of features missing
        remaining_numerical_features = len(numerical_features_missing)
        print(f"Numerical features remaining: {remaining_numerical_features}")

        # Select feature with most missing values
        feature = numerical_features_missing[0]
        print(feature)

        # Select target feature rows that aren't missing
        y_train = DATA[[feature]].dropna(axis=0, how='any')

        # Select training data, skip rows where the target feature isn't missing
        X_train = DATA[numerical_features].dropna(axis=1).iloc[list(y_train.index.values),:]

        # Select test data (rows with missing 'feature')
        X_test = DATA[pd.isnull(DATA[[feature]]).values][X_train.columns.values]

        # Fit and predict using selected model
        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train.values.ravel())
        yPred = regressor.predict(X_test).flatten().tolist()

        # Replace NaN with predicted values
        yToFill = pd.DataFrame(yPred, columns=[feature])

        # Combine full dataset with predicted values and convert the feature to float (so it can be used later)
        #DATA[feature] = DATA[feature].combine_first(yToFill)
        #DATA[DATA.isnull()] = yToFill
        DATA.loc[DATA[feature].isnull(), feature] = yPred
        DATA[feature] = pd.to_numeric(DATA[feature], errors='coerce')

        # Decrease 1 from remaining number of features that have NA values
        remaining_numerical_features -= 1
        print(' -Done')

    # Encode categorical variables
    # ----------------------------
    print("----------------------------")
    print("Encode categorical variables")
    print("----------------------------")
    from sklearn import preprocessing

    # Select categorical variables
    #categorical_features = full_data.select_dtypes(include=['object'])
    categorical_features = full_data.dtypes[full_data.dtypes == "object"].index

    # Apply cat.codes
    full_data[categorical_features] = full_data[categorical_features].apply(lambda x: x.astype('category').cat.codes)

    # Convert -1 values back to NaN
    full_data[full_data[categorical_features] < 0] = np.NaN

    # Predict Categorical variables
    # -----------------------------
    print("--------------------------------------")
    print("Start predicting categorical variables")
    print("--------------------------------------")
    while(remaining_categorical_features>0):
        # Get top missing features sorted that have more than 0 missing value
        categorical_features_missing = (DATA[categorical_features].isnull().sum()/DATA[categorical_features].isnull().count()).sort_values(ascending=False)
        categorical_features_missing = categorical_features_missing[categorical_features_missing>0].index

        # Number of features missing
        remaining_categorical_features = len(categorical_features_missing)
        print(f"Categorical features remaining: {remaining_categorical_features}")

        # Select feature with most missing values
        feature = categorical_features_missing[0]
        print(feature)

        # Select target feature rows that aren't missing
        y_train = DATA[[feature]].dropna(axis=0, how='any')

        # Select training data, skip rows where the target feature isn't missing
        X_train = DATA.dropna(axis=1).iloc[list(y_train.index.values),:]

        # Select test data (rows with missing 'feature')
        X_test = DATA[pd.isnull(DATA[[feature]]).values][X_train.columns.values]

        # Fit and predict using selected model
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train.values.ravel())
        yPred = classifier.predict(X_test).flatten().tolist()

        # Replace NaN with predicted values
        yToFill = pd.DataFrame(yPred, columns=[feature])

        # Combine full dataset with predicted values and convert the feature to float (so it can be used later)
        DATA.loc[DATA[feature].isnull(), feature] = yPred
        DATA[feature] = pd.to_numeric(DATA[feature], errors='coerce')

        # Decrease 1 from remaining number of features that have NA values
        remaining_categorical_features -= 1
        print(' -Done')

    return DATA, categorical_features

# Fill missing NaN
full_data, to_be_dummied = fillMissingAll()

# Add total area feature
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

# Get Dummies
# PS: Some not every categorical feature should be converted to a dummy variable. Some of them should be in order of quality for example
full_data = pd.get_dummies(full_data, columns=to_be_dummied, drop_first=True)
full_data.shape

# Separate full_data into training and test
n_train = train_data.shape[0]
X_train = full_data[:n_train]
X_test = full_data[n_train:]

# Determine the features most correlated to target
top_corr = X_train.assign(SalePrice=pd.Series(y_train).values).corr()
top_corr = top_corr[['SalePrice']].abs().sort_values(by='SalePrice', ascending=False)
top_corr = list(top_corr[top_corr>0.18][1:].dropna().index)

X_train = X_train[top_corr]
X_test = X_test[top_corr]


# Visualize correlation between features
# --------------------------------------
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(24,24))

# Compute the correlation matrix
corr = full_data.dropna().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap\n",
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=0, cbar_kws={"shrink": .5}, annot=False)


# Modelling
# -------
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from operator import itemgetter


# Load models
xgb = XGBRegressor(nthread = -1)
rfr = RandomForestRegressor(bootstrap= True, max_depth= 3, min_samples_leaf= 1000, min_samples_split= 5, n_estimators= 5)

# Cross_val_score
score = cross_val_score(rfr, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
print(score)


# Parameter tuning
# ----------------
from sklearn.grid_search import GridSearchCV
param_grid = {'bootstrap': [True, False],
               'max_depth': [3, 5, 7, 10, 20, None],
               'min_samples_leaf': [100, 200, 500, 1000, 1500, 3000],
               'min_samples_split': [2, 3, 5, 10],
               'n_estimators': [5, 10, 20, 40, 80, 200]}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=7).fit(X_train, y_train)

best_parameters = sorted([x[0:2] for x in grid_search.grid_scores_], key=itemgetter(1))[0]
print(best_parameters)
print(grid_search.best_params_)


rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)


# Submission
# ----------
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = y_pred
submission.to_csv('submission.csv',index=False)
