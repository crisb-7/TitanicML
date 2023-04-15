import joblib
import pandas as pd
from etl import data_pipeline

def main():
    test = pd.read_csv("./datasets/titanic_test.csv")
    passenger_id = test.PassengerId
    test = data_pipeline(test.drop(columns="PassengerId"))

    loaded_model = joblib.load("./models/gradient_boosting_cv10.sav")
    print(loaded_model.get_params())

    predictions = loaded_model.predict(test)
    predictions = pd.DataFrame({"PassengerId":passenger_id, "Survived":predictions})
    create_submission_file(predictions, 7, False)

def create_submission_file(submission, submission_number, submit_predictions):
    if submit_predictions:
        submission.to_csv(f"./submissions/submission_{submission_number}.csv", index=False)

if __name__ == "__main__":
    main()