from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np
import typer


def k_nearest(df: pd.DataFrame):
    """Predicts whether a team will win based on past performance"""
    neigh = KNeighborsClassifier(n_neighbors=3)
    # df["won"] = np.where(df["won_by"] > 0, 1, 0)
    neigh.fit(df[["avg_pass_yds_y", "avg_rush_yds_y"]], df["won_by"])
    print(neigh.score(df[["avg_pass_yds_y", "avg_rush_yds_y"]], df["won_by"]))


def linear_regression(df: pd.DataFrame):
    """Predicts how much the home team will win by"""
    reg = linear_model.LinearRegression()
    reg.fit(df[["avg_pass_yds_y", "avg_rush_yds_y"]], df["won_by"])
    print(reg.score(df[["avg_pass_yds_y", "avg_rush_yds_y"]], df["won_by"]))


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["pos"] == "QB"]
    df = df.drop(
        ["Surface", "Temperature", "Humidity", "Wind_Speed", "Roof", "OT"],
        axis=1,
    )
    df["is_home"] = df["team"] == df["home_team"]
    df["won_by"] = np.where(
        df["is_home"],
        df["home_score"] - df["vis_score"],
        df["vis_score"] - df["home_score"],
    )
    # Account for multiple quarterbacks in a game by adding their performances
    analyze_cols = ["pass_yds", "rush_yds"]
    for col in analyze_cols:
        df[f"tot_{col}"] = df.groupby(["game_id", "team"])[col].transform("sum")
        avg_row = (
            df.groupby("team")[f"tot_{col}"]
            .rolling(10, closed="left", min_periods=1)
            .mean()
        )
        avg_row = avg_row.rename(f"avg_{col}")
        avg_row.index = [x[1] for x in avg_row.index]
        df = df.merge(avg_row, left_index=True, right_index=True)
    df.drop_duplicates(["game_id", "team"], inplace=True)
    for col in analyze_cols:
        df = df[df[f"avg_{col}"].notna()]
    for col in analyze_cols:
        df[f"avg_{col}"] = np.where(df["is_home"], df[f"avg_{col}"], -df[f"avg_{col}"])
    cols = [f"avg_{col}" for col in analyze_cols]
    temp_df = df[["game_id"] + cols].groupby("game_id").agg("sum")
    df = df.merge(temp_df, on="game_id")
    df = df.loc[df["is_home"] == True]

    return df


def main(save: bool = False):
    df = pd.read_csv("data.csv")
    nfl = clean_df(df)
    linear_regression(nfl)
    k_nearest(nfl)
    if save:
        nfl.to_csv("export.csv")


if __name__ == "__main__":
    typer.run(main)
