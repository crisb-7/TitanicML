import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def main():
    train = pd.read_csv("./datasets/train.csv")

    x_train = train.drop(columns=["PassengerId", "Survived"])
    y_train = train.Survived
    
    model = find_best_gb_model(x_train, y_train)
    feature_importance = get_feature_importances(model, x_train)
    
    print(feature_importance.sort_values(by="Importance", ascending=False)[0:5])
    print("Train score: ", round(model.score(x_train, y_train), 4))
    
    filename = 'gradient_boosting_cv10.sav'
    joblib.dump(model, "./models/" + filename)


def find_best_gb_model(x, y):
    model = GradientBoostingClassifier(random_state=0)
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.5],
        "n_estimators": [25, 50, 100, 500, 1000, 2000],
        "subsample": [0.1, 0.25, 0.5, 0.75, 1],
        "criterion": ["friedman_mse", "squared_error"],
        "min_samples_split": [2, 3, 5, 10],
        "min_samples_leaf": [1, 2, 3], 
        "max_depth": [1, 2, 3, 4, 5],
        "min_impurity_decrease": [0.001, 0.01, 0.025, 0.05],
        "max_features": [None, "sqrt", "log2"]
    }

    n_folds = 10
    iters = 10
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=iters, cv=n_folds)
    search_results = random_search.fit(x, y)
    results_df = pd.DataFrame(search_results.cv_results_).sort_values("rank_test_score").drop(columns="params")
    print("Mean cross-val score:", results_df.mean_test_score)

    return search_results.best_estimator_

def get_feature_importances(model, data):
    return pd.DataFrame({"Feature":data.columns.to_list(), "Importance":model.feature_importances_})


if __name__ == "__main__":
    main()