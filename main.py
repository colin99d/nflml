from sklearn import linear_model
import pandas as pd
import numpy as np


def linear_regression(df: pd.DataFrame):
    reg = linear_model.LinearRegression()
    reg.fit(df[["tot_pass_yds", "tot_rush_yds"]], df["won_by"])
    print(reg.score(df[["tot_pass_yds", "tot_rush_yds"]], df["won_by"]))


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["pos"] == "QB"]
    df.drop(
        ["Surface", "Temperature", "Humidity", "Wind_Speed", "Roof", "OT"],
        inplace=True,
        axis=1,
    )
    df["won_by"] = np.where(
        df["team"] == df["home_team"],
        df["home_score"] - df["vis_score"],
        df["vis_score"] - df["home_score"],
    )
    # Account for multiple quarterbacks in a game by adding their performances
    for row in ["pass_yds", "rush_yds"]:
        df[f"tot_{row}"] = df.groupby(["game_id", "team"])[row].transform("sum")
        avg_row = (
            df.groupby("team")[f"tot_{row}"].rolling(10, closed="left", min_periods=1).mean()
        )
        print(avg_row)
    df.drop_duplicates(["game_id", "team"], inplace=True)
    return df


def main():
    df = pd.read_csv("data.csv")
    nfl = clean_df(df)
    nfl.to_csv("export.csv")
    linear_regression(nfl)


if __name__ == "__main__":
    main()
