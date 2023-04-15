import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


def remove_outliers(series, n):
    # n - determine magnitude of outlier (1.5 - normal, 3 - extreme)
    q3, q1 = series.quantile([0.75, 0.25])
    iqr = q3 - q1
    print("Límite de datos atípicos:", q3 + n*iqr)
    series = series.where(series < q3 + n*iqr)
    return series

def standard_scale(series):
    return (series - series.mean())/series.std()

def min_max_scale(series):
    return (series - series.min())/series.max()

def get_deck(data):
    # Transform Cabin column to a Deck dummy variables; obtain which Deck the cabin was in
    decks = ["A", "B", "C", "D", "E", "U"]
    for deck in decks:
        titanic_class = data.Cabin.str.contains(deck).astype(int)
        data["Deck_" + deck] = titanic_class

def fill_null(data):
    age_sample = data.Age[data.Age.notna()].sample(data.Age.isna().sum())
    data.Age.fillna(pd.Series(age_sample.to_list(), index=data.Age[data.Age.isna()].index), inplace=True)
    fare_sample = data.Fare[data.Fare.notna()].sample(data.Fare.isna().sum())
    data.Fare.fillna(pd.Series(fare_sample.to_list(), index=data.Fare[data.Fare.isna()].index), inplace=True)
    data.Cabin.fillna("U", inplace=True)

def scale_numeric(data, scaling):
    if scaling == "standard":
        data["Age"] = standard_scale(data.Age)
        data["Fare"] = standard_scale(data.Fare)
    elif scaling == "minmax":
        data["Age"] = min_max_scale(data.Age)
        data["Fare"] = min_max_scale(data.Fare)

def get_dummy_variables(data):
    data["Sex"] = pd.get_dummies(data.Sex)["female"]      # Turn sex into one-hot, 1 is female, 0 is male
    data[["Class1", "Class2", "Class3"]] = pd.get_dummies(data.Pclass)    # Ticket Class to dummy variables
    data[["Cherbourg", "Queenstown", "Southampton"]] = pd.get_dummies(data.Embarked)      # Embarked to dummy variables

def get_same_tickets(data):
    same_tickets = data.Ticket.value_counts()
    family_size = []
    for ticket in data.Ticket:
        family_size.append(same_tickets[ticket])
    data["SameTickets"] = family_size

def get_lone_travelers(data):
    alone = (data.SibSp < 1) & (data.Parch < 1)       # Generate Lone traveler variable
    data["Alone"] = alone.astype(int)

def box_cox_transform(data):
    data["Fare"], _ = stats.boxcox(data.Fare + 1e-2)
    data["Age"], _ = stats.boxcox(data.Age + 1e-2)

def get_family_size(data):
    data["FamSize"] = data.Parch + data.SibSp

def lognormal_transform(data):
    data["Fare"] = np.log(data.Fare + 1)

def get_titles(data):
    # titles = ["Mr.", "Mrs.", "Miss.", "Master."]
    titles = ["Master."]
    for title in titles:
        data[title.replace(".", "")] = data.Name.str.contains(title, regex=False).astype(int)
    return data

def perform_pca(data):
    cols = data.corr(numeric_only=True).columns
    dfs = data[cols]
    scaler = StandardScaler()
    dfs = scaler.fit_transform(dfs)
    dfs = pd.DataFrame(dfs, columns = cols)
    R = dfs.corr()
    lam, v = np.linalg.eig(R)
    M = np.array(dfs[cols])
    PCA = (v @ M.T)
    PCA = pd.DataFrame(PCA.T)
    return PCA


def data_pipeline(dataframe):

    # Call functions without redifining "dataframe"? Can inplace=True access the argument?
    fill_null(dataframe)

    # One-hot encoding of categorical data
    get_dummy_variables(dataframe)

    # Feature extraction
    get_lone_travelers(dataframe)
    get_family_size(dataframe)
    get_same_tickets(dataframe)
    get_deck(dataframe)
    get_titles(dataframe)

    # Numeric transformations
    lognormal_transform(dataframe)
    scale_numeric(dataframe, scaling="minmax")
    
    dataframe.drop(columns=["Pclass", "Name", "Ticket", "Cabin", "Embarked"], inplace=True)

    # dataframe = perform_pca(dataframe)

    return dataframe