import pandas as pd
from titanic_data_pipeline import data_pipeline

def main():
    df = pd.read_csv("./datasets/titanic_train.csv")
    train = data_pipeline(df)
    train.to_csv("./datasets/train.csv", index=False)


if __name__ == "__main__":
    main()