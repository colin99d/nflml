from typing import List, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np
import typer


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
    def __init__(self, analyze: Dict[str, Any], offense: pd.DataFrame):
        self.analyze = analyze
        self.offense = offense
        self.results = pd.DataFrame(columns=["name", "r2", "% correct"])
        self.k_nearest(self.offense)
        self.linear_regression(self.offense)
        self.results = self.results.set_index("name")
        print(self.results)

    def test_model(self, model, df: pd.DataFrame, name: str):
        columns = [f"net_avg_{x}" for x in self.analyze["offense"]]
        model.fit(df[columns], df["net_score"])
        r2 = model.score(df[columns], df["net_score"])
        df = pd.DataFrame([[name, r2, 0]], columns=["name", "r2", "% correct"])
        self.results = pd.concat([self.results, df])

    def k_nearest(self, df: pd.DataFrame, neighbors: int = 3):
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        self.test_model(neigh, df, "KNN")

    def linear_regression(self, df: pd.DataFrame):
        reg = linear_model.LinearRegression()
        self.test_model(reg, df, "LinReg")


def main(save: bool = False):
    analyze = {"offense": ["pass_yds", "rush_yds"], "defense": [], "kicking": []}
    data = Data(analyze)
    analyze = Analyze(analyze, data.offense)
    if save:
        data.offense.to_csv("export.csv")


if __name__ == "__main__":
    typer.run(main)
