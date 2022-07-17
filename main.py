from typing import List, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, svm
import pandas as pd
import numpy as np
import typer
from tqdm import tqdm


offense_unused = [
    "Surface",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Roof",
    "OT",
    "Team_abbrev",
    "Opponent_abbrev",
    "game_date",
    "player_id",
    "player",
]


defense_unused: List[str] = []
kicker_unused: List[str] = []


class Data:
    unused = {
        "offense": offense_unused,
        "defense": defense_unused,
        "kicking": kicker_unused,
    }

    def __init__(self, analyze):
        self.analyze = analyze
        self.offense = pd.DataFrame()
        self.defense = pd.DataFrame()
        self.kicking = pd.DataFrame()

        for item in ["offense", "defense", "kicking"]:
            df = pd.read_csv(f"{item}.csv")
            df = df.drop(self.unused[item], axis=1)
            if item == "offense":
                df = df[df["pos"] == "QB"]
            setattr(self, item, df)
        for item in self.analyze["offense"]:
            self.add_rolling(self.offense, item)
        self.clean_offense()

    def clean_offense(self):
        df = self.offense
        for column in self.analyze["offense"]:
            df = df[df[f"avg_{column}"].notna()]
        df = df.drop(["pos"], axis=1)
        df_home = df[df["team"] == df["home_team"]]
        df_away = df[df["team"] != df["home_team"]]
        df_away = df_away.drop(
            [
                "Vegas_Line",
                "Vegas_Favorite",
                "Over_Under",
                "vis_team",
                "home_team",
                "vis_score",
                "home_score",
            ],
            axis=1,
        )
        combined = df_home.merge(df_away, on="game_id", suffixes=["_home", "_away"])
        combined["net_score"] = combined["home_score"] - combined["vis_score"]
        for item in self.analyze["offense"]:
            combined[f"net_avg_{item}"] = (
                combined[f"avg_{item}_home"] - combined[f"avg_{item}_away"]
            )
        combined["vegas_favorite"] = np.where(
            combined["Vegas_Favorite"] == combined["home_team"], "HOME", "AWAY"
        )
        self.offense = combined

    def add_rolling(self, df: pd.DataFrame, column: str, rolling: int = 10):
        """Creates rolling avergae for given column. Does NOT include most recent row
        because we do not have this data beforehand"""
        tmp = df.copy(deep=True)
        tmp[f"tot_{column}"] = tmp.groupby(["game_id", "team"])[column].transform("sum")
        avg_row = (
            tmp.groupby("team")[f"tot_{column}"]
            .rolling(rolling, closed="left", min_periods=1)
            .mean()
        )
        avg_row = avg_row.rename(f"avg_{column}")
        avg_row.index = [x[1] for x in avg_row.index]
        tmp = tmp.merge(avg_row, left_index=True, right_index=True)
        tmp.drop_duplicates(["game_id", "team"], inplace=True)
        self.offense = tmp


class Analyze:
    models = ["k_nearest", "linear_regr", "decision_tree_regr", "vector_class_regr"]

    def __init__(self, analyze: Dict[str, Any], offense: pd.DataFrame, runs: int = 100):
        self.analyze = analyze
        selections = {"base": [f"net_avg_{x}" for x in self.analyze["offense"]]}
        self.offense = offense
        self.results = pd.DataFrame(columns=["name"] + list(selections.keys()))
        self.runs = runs

        # Run models
        for key, val in tqdm(selections.items(), position=0):
            for model in tqdm(self.models, position=1):
                getattr(self, model)(self.offense, val, key)

        self.results = self.results.set_index("name")
        print(f"Average percent correct for {self.runs} runs.")
        print(self.results)

    def test_model(
        self, model, df: pd.DataFrame, model_name: str, X: List[str], X_name: str
    ):
        # See how model does at vegas
        scores = []
        for _ in range(self.runs):
            x_train, x_test, y_train, _ = train_test_split(
                df[X], df["net_score"], test_size=0.2
            )
            model.fit(x_train, y_train)
            x_test["preds"] = model.predict(x_test)
            temp = df[["Vegas_Line", "net_score", "vegas_favorite"]]
            combined = x_test.merge(temp, left_index=True, right_index=True)
            combined["vegas"] = combined["Vegas_Line"] * np.where(
                combined["vegas_favorite"] == "HOME", -1, 1
            )
            combined["bet"] = np.where(
                combined["vegas_favorite"] == "HOME",
                np.where(combined["preds"] > combined["vegas"], "HOME", "AWAY"),
                np.where(combined["vegas"] > combined["preds"], "HOME", "AWAY"),
            )
            combined["success"] = np.where(
                combined["bet"] == "HOME",
                combined["net_score"] > combined["vegas"],
                combined["vegas"] > combined["net_score"],
            )
            score = combined["success"].value_counts(normalize=True).at[True]
            scores.append(score)
        df = pd.DataFrame([[model_name, np.mean(scores)]], columns=["name", X_name])
        self.results = pd.concat([self.results, df])

    def k_nearest(self, df: pd.DataFrame, X: List[str], name: str, neighbors: int = 3):
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        self.test_model(neigh, df, "KNN", X, name)

    def linear_regr(self, df: pd.DataFrame, X: List[str], name: str):
        reg = linear_model.LinearRegression()
        self.test_model(reg, df, "LinReg", X, name)

    def decision_tree_regr(self, df: pd.DataFrame, X: List[str], name: str):
        clf = tree.DecisionTreeRegressor()
        self.test_model(clf, df, "TreeReg", X, name)

    def vector_class_regr(self, df: pd.DataFrame, X: List[str], name: str):
        regr = svm.SVR()
        self.test_model(regr, df, "SVReg", X, name)


def main(save: bool = False, runs: int = 100):
    analyze = {"offense": ["pass_yds", "rush_yds"], "defense": [], "kicking": []}
    data = Data(analyze)
    analyze = Analyze(analyze, data.offense, runs)
    if save:
        data.offense.to_csv("export.csv")


if __name__ == "__main__":
    typer.run(main)
