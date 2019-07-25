import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


def my_confusion_matrix(labs, preds):
    classes = np.unique(labs).astype(int)
    cm = confusion_matrix(labs, preds, classes)
    return pd.DataFrame(
        cm,
        columns=pd.MultiIndex.from_tuples([("pred", i) for i in classes]),
        index=pd.MultiIndex.from_tuples([("true", i) for i in classes]),
    )


def plot_confusion_matrices(y_train, pred_train, y_test, pred_test):
    print("train")
    display(my_confusion_matrix(y_train, pred_train))
    print("\n\ntest")
    display(my_confusion_matrix(y_test, pred_test))


def my_cross_validate(algo, X, y, scoring, n_splits=3, random_state=1):
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    scores = cross_validate(algo, X, y, cv=kf, scoring=scoring, return_train_score=True)
    # Take the mean of the scores (we have one score per fold)
    for scorer in scoring:
        print(
            f"{scorer} --> "
            f"Train: {round(scores[f'train_{scorer}'].mean(), 3)}, "
            f"Test: {round(scores[f'test_{scorer}'].mean(), 3)},"
        )


def grid_search_algo(
    algo,
    param_grid,
    X,
    y,
    cv=StratifiedKFold(10, shuffle=True),
    n_jobs=3,
    scoring=["recall", "precision", "f1"],
    refit="f1",
):
    gs = GridSearchCV(
        algo,
        param_grid,
        iid=False,
        cv=cv,
        return_train_score=True,
        verbose=1,
        n_jobs=n_jobs,
        scoring=scoring,
        refit=refit,
    )
    search = gs.fit(X, y)
    return search


def analyse_grid_search(result):
    df_result = pd.DataFrame(result.cv_results_)
    scores_columns = [
        "mean_test_recall",
        "mean_test_precision",
        "mean_test_fbeta",
        "std_test_fbeta",
        "mean_train_recall",
        "mean_train_precision",
        "mean_train_fbeta",
    ]
    params_columns = [col for col in df_result.columns if col.startswith("param_")]
    times_columns = ["mean_fit_time"]
    columns = scores_columns + params_columns + times_columns
    display(
        df_result.sort_values(
            ["mean_test_fbeta", "mean_fit_time"], ascending=[False, True]
        )[columns].T
    )
    display(df_result[scores_columns].describe().loc[["max", "min"]].T)
