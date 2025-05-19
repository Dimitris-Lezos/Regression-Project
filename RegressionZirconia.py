import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import category_encoders as ce


def read_and_preprocess() -> pd.DataFrame:
    data = pd.read_csv('train.csv')
    #### 1. Check for missing values
    data.head()
    data.shape
    features = pd.DataFrame({
        'Count': data.count(),
        'Unique': data.nunique(),
        'Unique Percent': round(((data.nunique()/ data.count()) * 100),2),
        'Null Count': data.isnull().sum(),
        'Null Percent': data.isna().sum()* 100 / data.shape[0],
        'Data Type': data.dtypes
    })
    print(features)
    print(data.describe())
    #### 2. Check for duplicates
    duplicates = data.duplicated()
    print('Duplicates:', duplicates.loc[duplicates == True])
    # No Duplicates
    return data


def encode_with_ordinalencoder(data: pd.DataFrame) -> pd.DataFrame:
    #cuts = [['Fair'], ['Good'], ['Very Good'], ['Premium'], ['Ideal'], ['Fair']]
    ## 3.a Ordinal Encoding
    ordinal_columns = data[['cut', 'color', 'clarity']]
    # Worst -> Best: 0, 1, 2...
    encoder = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                         ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                         ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']])
    # Fit and transform the data
    data[['cut', 'color', 'clarity']] = encoder.fit_transform(ordinal_columns)
    return data


def encode_with_helmertencoding(data: pd.DataFrame) -> pd.DataFrame:
    # 3.b Helmert Encoding
    # Very poor results
    # Instantiate the encoder
    encoder = ce.HelmertEncoder(cols=['cut', 'color', 'clarity'])
    # Fit and transform the data
    df_encoded = encoder.fit_transform(data[['cut', 'color', 'clarity']])
    price = data[['price']]
    data.drop(columns=['price'], inplace=True)
    data = data.join(df_encoded)
    data = data.join(price)
    data.drop(columns=['cut', 'intercept', 'color', 'clarity'], inplace=True)
    # Print the encoded data
    print(data.head())
    return data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    # Do not normalize price
    price = data[['price']]
    data = pd.DataFrame(MinMaxScaler().fit_transform(data.iloc[:, :-1]), columns=data.columns[:-1])
    data = data.join(price)
    try:
        data.drop(columns=['id'], inplace=True)
    except:
        pass
    return data


def split_to_train_and_test(data: pd.DataFrame, target=None, random_state=42) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    if target is None:
        target = data.iloc[:, -1:]
        data = data.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.3,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test


def replace_price_with_price_per_carat(data: pd.DataFrame) -> pd.DataFrame:
    data['price'] = data['price'] / data['carat']
    return data


def verify_split(x_train, x_test, show_plots=False):
    try:
        print(x_train['cut'].value_counts(normalize=True), x_test['cut'].value_counts(normalize=True))
        print(x_train['color'].value_counts(normalize=True), x_test['color'].value_counts(normalize=True))
        print(x_train['clarity'].value_counts(normalize=True), x_test['clarity'].value_counts(normalize=True))
    except Exception:
        pass
    if show_plots:
        # Also show in plots
        plt.figure(1)
        plt.plot(x_train['cut'].value_counts(normalize=True, sort=False).sort_index() + 0.02)
        plt.plot(x_test['cut'].value_counts(normalize=True, sort=False).sort_index())
        plt.figure(2)
        plt.plot(x_train['color'].value_counts(normalize=True, sort=False).sort_index() + 0.02)
        plt.plot(x_test['color'].value_counts(normalize=True, sort=False).sort_index())
        plt.figure(3)
        plt.plot(x_train['clarity'].value_counts(normalize=True, sort=False).sort_index() + 0.02)
        plt.plot(x_test['clarity'].value_counts(normalize=True, sort=False).sort_index())
        plt.show()


def show_correlation_heatmap(data: pd.DataFrame) -> None:
    print(data.describe())
    corr_matrix = data.corr()
    f, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                annot_kws={'size': 8},
                cmap="Spectral_r")
    plt.show()


def remove_columns(data: pd.DataFrame, columns=['x', 'y', 'z']) -> pd.DataFrame:
    data.drop(columns=columns, inplace=True)
    return data


def show_normalized_boxplot(data: pd.DataFrame):
    scaler = MinMaxScaler()
    box_data = data
    try:
        box_data = data.drop(columns=['price'])
    except Exception:
        pass
    box_data = pd.DataFrame(scaler.fit_transform(box_data), index=box_data.index, columns=box_data.columns)
    sns.boxplot(data=box_data)
    plt.show()


def print_prediction_results(label: str, y_test: pd.DataFrame, y_pred: pd.DataFrame):
    # calculate mse
    mse = mean_squared_error(y_test['price'], y_pred)
    mae = mean_absolute_error(y_test['price'], y_pred)
    print('------------------------')
    print(label, 'MSE=', round(mse, 2), '(* Mean Square Error)')
    print(label, 'MAE=', round(mae, 2), '(* Mean Absolute Error)')
    print(label, ' R2=', r2_score(y_test['price'], y_pred))


def linear_regression(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, label='') -> pd.DataFrame:
    # Create the model
    linear_regression = LinearRegression()
    # training the linear model
    linear_regression.fit(x_train, y_train['price'])
    # Predict
    y_pred = linear_regression.predict(x_test)
    print_prediction_results('Linear Regression'+label, y_test, y_pred)
    return y_pred


def linear_regression_for_each_column(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    for col in x_train.columns:
        linear_regression(x_train[[col]], y_train, x_test[[col]], y_test, ' column \''+col+'\'')


def plot_regression(x_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame):
    plt.figure(1)
    plt.title('Linear Regression')
    plt.ylabel('yyy')
    plt.xlabel('Input feature=carat')
    plt.scatter(x_test[:100]['carat'], y_test[:100]['price'], color='black')
    plt.scatter(x_test[:100]['carat'], y_pred[:100], color='blue')
    plt.xticks(())
    plt.yticks(())
    plt.show()


def apply_PCA(data: pd.DataFrame, n_components=6) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    prices = data.iloc[:,-1:]
    pca = PCA(n_components=n_components)
    fit_data = pd.DataFrame(pca.fit_transform(data.iloc[:, :-1]))
    print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
    return split_to_train_and_test(fit_data, prices)


def polynomial_regression(degrees: range, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, label='') -> pd.DataFrame:
    for degree in degrees:
        # Create the model
        linear_regression = LinearRegression()
        poly = PolynomialFeatures(degree=degree)
        # training the linear model
        linear_regression.fit(poly.fit_transform(x_train), y_train['price'])
        # Predict
        y_pred = linear_regression.predict(poly.fit_transform(x_test))
        print_prediction_results('Polynomial Regression degree='+str(degree)+label, y_test, y_pred)
    return None


def polynomial_regression_for_each_column(degrees: range, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    for degree in degrees:
        for col in x_train.columns:
            polynomial_regression(range(degree, degree+1), x_train[[col]], y_train, x_test[[col]], y_test, ' column \''+col+'\'')
    return None


def ridge_regression(degree: int, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    poly = PolynomialFeatures(degree=degree)
    ridge_regression = Ridge(alpha=0.1)
    ridge_regression.fit(poly.fit_transform(x_train), y_train['price'])
    y_pred = ridge_regression.predict(poly.fit_transform(x_test))
    print_prediction_results('Ridge Regression degree=' + str(degree), y_test, y_pred)
    return y_pred


def lasso_regression(degree: int, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    poly = PolynomialFeatures(degree=degree)
    lasso_regression = Lasso(alpha=0.1)
    lasso_regression.fit(poly.fit_transform(x_train), y_train['price'])
    y_pred = lasso_regression.predict(poly.fit_transform(x_test))
    print_prediction_results('Lasso Regression degree=' + str(degree), y_test, y_pred)
    return y_pred


def backward_stepwise_polynomial_regression(degrees: range, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, columns=['depth', 'cut', 'table', 'clarity', 'color']):
    x_train_drop = x_train.copy()
    x_test_drop = x_test.copy()
    for column in columns:
        x_train_drop.drop(columns=[column], inplace=True)
        x_test_drop.drop(columns=[column], inplace=True)
        polynomial_regression(degrees, x_train_drop, y_train, x_test_drop, y_test, str(list(x_train_drop.columns)))


def nn_regresion(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
    from tensorflow.keras import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanSquaredError

    # Creating model using the Sequential in tensorflow
    model = Sequential([
        Dense(100, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(200, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(50, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
    ])
    # loss function
    msle = MeanSquaredLogarithmicError()
    mse = MeanSquaredError()
    model.compile(
        loss=msle,
        optimizer=Adam(learning_rate=0.01),
        # metrics=[msle],
        metrics=[mse],
    )
    # train the model
    history = model.fit(
        x_train.values,
        y_train['price'].values,
        epochs=20,
        batch_size=300,
        validation_split=0.2,
    )
    y_pred = model.predict(x_test)
    print_prediction_results('Neural Network Regression', y_test, y_pred)


def encode_ordinal_plus(x_train: pd.DataFrame, x_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    #### 4.5 Find splits for
    ##print(x_train[['cut', 'price_carat']].groupby('cut').mean())
    #      price_carat
    # cut
    # 0.0  4178.218294
    # 1.0  4075.386070
    # 2.0  4089.093958
    # 3.0  4371.447664
    # 4.0  3836.810796
    x_train.loc[x_train['cut'] == 0.0, 'cut'] = 4178.218294
    x_train.loc[x_train['cut'] == 1.0, 'cut'] = 4075.386070
    x_train.loc[x_train['cut'] == 2.0, 'cut'] = 4089.093958
    x_train.loc[x_train['cut'] == 3.0, 'cut'] = 4371.447664
    x_train.loc[x_train['cut'] == 4.0, 'cut'] = 3836.810796
    x_test.loc[x_test['cut'] == 0.0, 'cut'] = 4178.218294
    x_test.loc[x_test['cut'] == 1.0, 'cut'] = 4075.386070
    x_test.loc[x_test['cut'] == 2.0, 'cut'] = 4089.093958
    x_test.loc[x_test['cut'] == 3.0, 'cut'] = 4371.447664
    x_test.loc[x_test['cut'] == 4.0, 'cut'] = 3836.810796
    # print(x_train[['cut', 'price']].groupby('cut').mean())
    ##print(x_train[['color', 'price_carat']].groupby('color').mean())
    #        price_carat
    # color
    # 0.0    4074.421647
    # 1.0    4242.263410
    # 2.0    4198.260033
    # 3.0    4243.583535
    # 4.0    4064.675889
    # 5.0    3722.898515
    # 6.0    3757.711006
    x_train.loc[x_train['color'] == 0.0, 'color'] = 4074.421647
    x_train.loc[x_train['color'] == 1.0, 'color'] = 4242.263410
    x_train.loc[x_train['color'] == 2.0, 'color'] = 4198.260033
    x_train.loc[x_train['color'] == 3.0, 'color'] = 4243.583535
    x_train.loc[x_train['color'] == 4.0, 'color'] = 4064.675889
    x_train.loc[x_train['color'] == 3.0, 'color'] = 3722.898515
    x_train.loc[x_train['color'] == 4.0, 'color'] = 3757.711006
    x_test.loc[x_test['color'] == 0.0, 'color'] = 4074.421647
    x_test.loc[x_test['color'] == 1.0, 'color'] = 4242.263410
    x_test.loc[x_test['color'] == 2.0, 'color'] = 4198.260033
    x_test.loc[x_test['color'] == 3.0, 'color'] = 4243.583535
    x_test.loc[x_test['color'] == 4.0, 'color'] = 4064.675889
    x_test.loc[x_test['color'] == 3.0, 'color'] = 3722.898515
    x_test.loc[x_test['color'] == 4.0, 'color'] = 3757.711006
    # print(x_train[['color', 'price']].groupby('color').mean())
    ##print(x_train[['clarity', 'price_carat']].groupby('clarity').mean())
    #          price_carat
    # clarity
    # 2.0      3052.792088
    # 3.0      4186.055291
    # 4.0      3987.874720
    # 5.0      4158.849531
    # 6.0      4130.439523
    # 7.0      3946.810601
    # 8.0      3442.975768
    # 9.0      3673.568498
    x_train.loc[x_train['clarity'] == 2.0, 'clarity'] = 3052.792088
    x_train.loc[x_train['clarity'] == 3.0, 'clarity'] = 4186.055291
    x_train.loc[x_train['clarity'] == 4.0, 'clarity'] = 3987.874720
    x_train.loc[x_train['clarity'] == 5.0, 'clarity'] = 4158.849531
    x_train.loc[x_train['clarity'] == 6.0, 'clarity'] = 4130.439523
    x_train.loc[x_train['clarity'] == 7.0, 'clarity'] = 3946.810601
    x_train.loc[x_train['clarity'] == 8.0, 'clarity'] = 3442.975768
    x_train.loc[x_train['clarity'] == 9.0, 'clarity'] = 3673.568498
    x_test.loc[x_test['clarity'] == 2.0, 'clarity'] = 3052.792088
    x_test.loc[x_test['clarity'] == 3.0, 'clarity'] = 4186.055291
    x_test.loc[x_test['clarity'] == 4.0, 'clarity'] = 3987.874720
    x_test.loc[x_test['clarity'] == 5.0, 'clarity'] = 4158.849531
    x_test.loc[x_test['clarity'] == 6.0, 'clarity'] = 4130.439523
    x_test.loc[x_test['clarity'] == 7.0, 'clarity'] = 3946.810601
    x_test.loc[x_test['clarity'] == 8.0, 'clarity'] = 3442.975768
    x_test.loc[x_test['clarity'] == 9.0, 'clarity'] = 3673.568498
    # Update these values to the original data
    # print(x_train.head())
    # print(x_test.head())
    # New Heatmap
    plt.figure(20)
    corr_matrix = x_train.corr()
    f, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                annot_kws={'size': 8},
                cmap="Spectral_r")
    plt.show()
    return x_train, x_test


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    # Normalize
    # Show Boxplot
    show_normalized_boxplot(data)
    # Keep carat < 0.54, 0.39 < depth < 0.62, 0.135 < table < 0.41
    no_outliers = data.copy()
    no_outliers = data[data['carat'] <= 0.54]
    no_outliers = data[data['depth'] <= 0.62]
    no_outliers = data[data['depth'] >= 0.39]
    no_outliers = data[data['table'] <= 0.41]
    no_outliers = data[data['table'] >= 0.135]
    return no_outliers

#################################################################################
# Preprocess data:
#### 1. Check for missing values
#### 2. Check for duplicates
data = read_and_preprocess()
#### 3. Encode Ordinal features:
data = encode_with_ordinalencoder(data)
#### 3.5 Remove the id column
data.drop(columns=['id', 'depth'], inplace=True)
data = remove_columns(data)
data.to_csv('cubic_zirconia_train.csv')
#exit(0)
#### 4. Normalize the data in the numerical columns
data = normalize_data(data)
#### 5. Split between training and test data
x_train, x_test, y_train, y_test = split_to_train_and_test(data)
#### 6. Verify that the split gives similar distributions between train and test on the Ordinal columns
verify_split(x_train, x_test, show_plots=True)
#### 5. Investigate the correlation of the data
show_correlation_heatmap(data)
#### 7. We see that carat, x, y, z are completely correlated, so we remove x, y & z features
data = remove_columns(data)
x_train = remove_columns(x_train)
x_test = remove_columns(x_test)
#### 8. Let's see the correlation again
show_correlation_heatmap(data)
#############################################################################################
# Price and carat are very closely correlated as expected we will also latter
# try to train for Price / carat
#############################################################################################
#### 9. Show Boxplots
show_normalized_boxplot(data)
#### 10. Do a simple linear regression for each column
linear_regression_for_each_column(x_train, y_train, x_test, y_test)
#### 11. Do a simple linear regression for all columns
y_pred = linear_regression(x_train, y_train, x_test, y_test)
#### 12. Plot regression
plot_regression(x_test, y_test, y_pred)
#### 13. Polynomial linear regression for each column
polynomial_regression_for_each_column(range(2,3), x_train, y_train, x_test, y_test)
#### 14. Polynomial linear regression
degrees = range(2,8)
polynomial_regression(degrees, x_train, y_train, x_test, y_test)
#### 15. Ridge around the best polynomials
for degree in range(2,6):
    ridge_regression(degree, x_train, y_train, x_test, y_test)
#### 16. Lasso around the best polynomials
for degree in range(2,6):
    lasso_regression(degree, x_train, y_train, x_test, y_test)
#### 17. Backward Stepwise Polynomial regression
backward_stepwise_polynomial_regression(degrees, x_train, y_train, x_test, y_test)
#### 18. Backward Stepwise Polynomial regression by PCA n_components
x_train_pca, x_test_pca, y_train_pca, y_test_pca = apply_PCA(data)
print("PCA, Ridge, Polynomial Regression: Backward Stepwise Polynomial regression")
degrees = range(2,6)
polynomial_regression(degrees, x_train_pca, y_train_pca, x_test_pca, y_test_pca)
columns = list(x_train_pca.columns)
backward_stepwise_polynomial_regression(degrees, x_train_pca, y_train_pca, x_test_pca, y_test_pca, columns)
#### 19. NN regression
nn_regresion(x_train, y_train, x_test, y_test)
#######################################################
#### 20. Best of class
#### Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']
print('------------------------')
print("Best of class: Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Try additional cases:
#### 21. Remove outliers
x_train = remove_outliers(x_train)
y_train = y_train.loc[x_train.index]
print('------------------------')
print("Best of class: NO OUTLIERS, Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Try additional cases
#### 22. Encode Ordinal features Helmert Encoding
data = read_and_preprocess()
data.drop(columns=['id'], inplace=True)
data = remove_columns(data)
data = encode_with_helmertencoding(data)
x_train, x_test, y_train, y_test = split_to_train_and_test(data)
print('------------------------')
print("Best of class: Helmert Encoding, Ridge, Polynomial Regression degree=4['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(4, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Try additional cases
#### 23. Encode Ordinal features Based on Price/Carat
data = read_and_preprocess()
data = encode_with_ordinalencoder(data)
data.drop(columns=['id'], inplace=True)
data = remove_columns(data)
x_train, x_test, y_train, y_test = split_to_train_and_test(data)
x_train, x_test = encode_ordinal_plus(x_train, x_test)
print('------------------------')
print("Best of class: Ordinal with Price/Carat, Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Try additional cases
#### 24. Convert price -> price / carat
data = read_and_preprocess()
data['price'] = data['price']/data['carat']
data = encode_with_ordinalencoder(data)
data.drop(columns=['id'], inplace=True)
data = remove_columns(data)
x_train, x_test, y_train, y_test = split_to_train_and_test(data)
print('------------------------')
print("Best of class: Price/Carat, Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
print("Best of class: Price/Carat, Ridge, Polynomial Regression degree=5['carat', 'cut', 'color', 'clarity', 'table']")
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Try additional cases
#### 25. Convert price -> price / carat, remove 'carat'
data = read_and_preprocess()
data['price'] = data['price']/data['carat']
data = encode_with_ordinalencoder(data)
data.drop(columns=['id'], inplace=True)
data = remove_columns(data)
data.drop(columns=['carat'], inplace=True)
x_train, x_test, y_train, y_test = split_to_train_and_test(data)
print('------------------------')
print("Best of class: Price/Carat, Ridge, Polynomial Regression degree=5['cut', 'color', 'clarity', 'table']")
ridge_regression(3, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
ridge_regression(4, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
#######################################################
#### Verify Best Results with other splits
####
for n in [4033234345, 2270196945, 2084006233, 3245149790, 2568624065, 245629106, 4291163709, 2388055995, 743943979, 119627208]:
    data = read_and_preprocess()
    data = encode_with_ordinalencoder(data)
    data.drop(columns=['id'], inplace=True)
    data = remove_columns(data)
    data = normalize_data(data)
    #### 5. Split between training and test data
    x_train, x_test, y_train, y_test = split_to_train_and_test(data, random_state=n)
    polynomial_regression(range(6,7), x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
for n in [4033234345, 2270196945, 2084006233, 3245149790, 2568624065, 245629106, 4291163709, 2388055995, 743943979, 119627208]:
    data = read_and_preprocess()
    data = encode_with_ordinalencoder(data)
    data.drop(columns=['id'], inplace=True)
    data = remove_columns(data)
    #data = normalize_data(data)
    #### 5. Split between training and test data
    x_train, x_test, y_train, y_test = split_to_train_and_test(data, random_state=n)
    ridge_regression(5, x_train.drop(columns=['depth']), y_train, x_test.drop(columns=['depth']), y_test)
exit(0)
