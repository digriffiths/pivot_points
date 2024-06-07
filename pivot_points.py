import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import multiprocessing as mp
from itertools import product
from typing import Tuple, List

plt.style.use("seaborn-v0_8")

class PivotPointStrategy:
    def __init__(self, dataframe: pd.DataFrame, window: int, days_high: int, stoploss: float):
        """
        Initialize the PivotPointStrategy class with the given parameters.
        
        Args:
        dataframe (pd.DataFrame): The input data frame containing the market data.
        window (int): The rolling window size for calculating pivot points.
        days_high (int): The number of consecutive days a value must be the highest to be considered a pivot point.
        stoploss (float): The stop loss percentage to limit losses in the strategy.
        """
        self.dataframe = dataframe
        self.window = window
        self.days_high = days_high
        self.stoploss = stoploss

    def find_pivs(self) -> pd.DataFrame:
        """
        Identify pivot points in the dataframe based on the initialized parameters.
        
        Returns:
        pd.DataFrame: The modified dataframe with pivot points established.
        """
        data = self.dataframe.copy()
        if self.window >= self.days_high:
            data["creturns_max"] = (
                data["creturns"].rolling(self.window + 1, min_periods=1).max()
            )
            data["prev_max"] = data["creturns_max"].shift(1)
            data["counter"] = (
                (data["creturns_max"] == data["prev_max"]).astype(int).cumsum()
            )
            data["reset_counter"] = (
                (data["creturns_max"] != data["prev_max"]).astype(int).cumsum()
            )
            data["counter"] -= data.groupby("reset_counter")["counter"].transform("min")

            data["pivot_points_established"] = np.where(
                data["counter"] == self.days_high, data["creturns_max"], np.nan
            )

            self.dataframe = data
            return data

    def process_pivot_points(self) -> pd.DataFrame:
        """
        Process the established pivot points and update the dataframe with their first occurrence.
        
        Returns:
        pd.DataFrame: The updated dataframe with pivot points processed.
        """
        pivot_points = self.dataframe["pivot_points_established"].dropna().unique()
        index: List[Tuple[float, int]] = [
            (pivot, idx)
            for pivot in pivot_points
            for idx in self.dataframe[self.dataframe.creturns == pivot].index
        ]
        df = pd.DataFrame(index, columns=["pivot_first_hit", "index_value"]).set_index(
            "index_value"
        )
        self.dataframe = self.dataframe.join(df["pivot_first_hit"])

        return self.dataframe

    def plot_pivots(self, line_length: int = 50) -> None:
        """
        Plot the pivot points on a graph.
        
        Args:
        line_length (int): The length of the line to extend the pivot point visualization.
        """
        dates = []
        pivots = []

        for datetime, row in self.dataframe.iterrows():
            value = row["pivot_first_hit"]
            dates.append(datetime)
            dates.append(datetime + timedelta(days=line_length))
            pivots.append(value)
            pivots.append(value)

        fig = plt.figure(figsize=(12, 8))
        plt.plot(dates, pivots, linestyle="dotted", label="Pivot Points", color="blue")
        plt.plot(
            self.dataframe.index,
            self.dataframe.creturns,
            label="creturns",
            color="black",
            linewidth=0.5,
        )
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.title(
            f"BTC - Pivot Points \n\n Window = {self.window}, Days High = {self.days_high}"
        )
        plt.legend()
        plt.show()

    def run_strat(self) -> pd.DataFrame:
        """
        Execute the trading strategy based on pivot points and stop loss criteria.
        
        Returns:
        pd.DataFrame: The dataframe with strategy execution results.
        """
        data = self.dataframe.copy()

        # Calculate pivot above and below
        pivot = []
        next_higher_piv = []
        next_lower_piv = []
        for index, row in data.iterrows():
            pivot.append(row["pivot_points_established"])
            current_price = row["creturns"]
            higher_pivots = [p for p in pivot if p > current_price]
            lower_pivots = [p for p in pivot if p < current_price]
            next_higher_piv.append(min(higher_pivots, default=np.nan))
            next_lower_piv.append(max(lower_pivots, default=np.nan))
        data["next_higher_piv"] = next_higher_piv
        data["next_lower_piv"] = next_lower_piv

        # Position when price crosses pivot
        data["position"] = np.where(
            data.creturns > data.next_higher_piv.shift(1), 1, np.nan
        )
        data["position"] = np.where(
            data.creturns < data.next_lower_piv.shift(1), -1, data["position"]
        )

        # Stop loss
        if self.stoploss > 0:
            data["position_chng"] = (data.position.ffill().diff() == 2) | (
                data.position.ffill().diff() == -2
            )
            data["strategy"] = data["position"].ffill().shift(1) * data["returns"]
            data["trade_ret"] = data["strategy"].where(~data["position_chng"], 0)
            trade_ret_lst = [0]
            for index, row in data.iterrows():
                cumsum = trade_ret_lst[-1] + row.strategy
                trade_ret_lst.append(cumsum if not row.position_chng else 0)
            data["trade_ret"] = trade_ret_lst[1:]
            data["position"] = np.where(
                data["trade_ret"] < -self.stoploss / 100, 0, data["position"]
            )

        # Forward fill position
        data["position"] = data.position.ffill()

        # Calculate cumulative returns
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        # Calculate prices for plot
        data["long_price"] = data.creturns[
            (data.position.diff() == 1) & (data.position > 0)
            | (data.position.diff() == 2) & (data.position > 0)
        ]
        data["short_price"] = data.creturns[
            (data.position.diff() == -1) & (data.position < 0)
            | (data.position.diff() == -2) & (data.position < 0)
        ]
        data["neutral_price"] = data.creturns[
            (data.position.diff() != 0) & (data.position == 0)
        ]

        self.dataframe = data

        return data

    def plot_strat(self) -> None:
        """
        Plot the strategy execution results on a graph.
        """
        dates = []
        pivots = []

        for datetime, row in self.dataframe.iterrows():
            value = row["pivot_first_hit"]
            dates.append(datetime)
            dates.append(datetime + timedelta(days=30))
            pivots.append(value)
            pivots.append(value)

        fig, ax1 = plt.subplots()
        fig.set_figwidth(12)
        fig.set_figheight(8)

        ax2 = ax1.twinx()
        x = self.dataframe.index
        ax1.plot(
            x, self.dataframe.creturns, label="creturns", color="black", linewidth=0.5
        )
        ax1.plot(
            x, self.dataframe.cstrategy, label="cstrategy", color="teal", linewidth=0.5
        )

        ax1.scatter(
            self.dataframe.index,
            self.dataframe.long_price,
            label="Long",
            marker="^",
            color="green",
            alpha=1,
        )
        ax1.scatter(
            self.dataframe.index,
            self.dataframe.short_price,
            label="Short",
            marker="v",
            color="red",
            alpha=1,
        )
        ax1.scatter(
            self.dataframe.index,
            self.dataframe.neutral_price,
            label="neutral",
            marker="o",
            color="blue",
            alpha=1,
        )
        ax1.plot(dates, pivots, linestyle="dotted", label="Pivot Points")

        ax1.legend()
        ax1.set_xlabel("Date")
        ax2.set_ylabel("Position")
        ax1.set_ylabel("Return")
        ax1.set_title(
            f"BTC - Pivot Points Strategy \n\n Window = {self.window}, Days High = {self.days_high}, Stop loss = {self.stoploss}%"
        )

        plt.show()

    def optimize(self, window: int, days_high: int, stoploss: float) -> pd.DataFrame:
        """
        Optimize the strategy parameters and return the results.
        
        Args:
        window (int): The rolling window size for calculating pivot points.
        days_high (int): The number of consecutive days a value must be the highest to be considered a pivot point.
        stoploss (float): The stop loss percentage to limit losses in the strategy.
        
        Returns:
        pd.DataFrame: A dataframe containing the optimization results.
        """
        # Update strategy parameters
        self.window = window
        self.days_high = days_high
        self.stoploss = stoploss

        # Calculate pivot points and run strategy
        self.find_pivs()
        self.run_strat()

        # Attempt to extract the last strategy return value
        try:
            strat_ret = self.dataframe['cstrategy'].iloc[-1]
            result = pd.DataFrame({
                "window": [window],
                "days_high": [days_high],
                "stoploss": [stoploss],
                "strat_ret": [strat_ret],
            })
        except IndexError as e:
            # Log the error and return an empty DataFrame if there's an issue
            print(f"Failed to extract strategy return: {e}")
            return pd.DataFrame()

        return result

    def run_optimization(self, window_range: Tuple[int, int], days_high_range: Tuple[int, int], stoploss_range: Tuple[int, int]) -> pd.DataFrame:
        """
        Run optimization over a range of parameters and return the best results.
        
        Args:
        window_range (Tuple[int, int]): The range of window sizes to test.
        days_high_range (Tuple[int, int]): The range of days high values to test.
        stoploss_range (Tuple[int, int]): The range of stop loss percentages to test.
        
        Returns:
        pd.DataFrame: A dataframe containing the best optimization results.
        """
        try:
            cpus = mp.cpu_count()
        except NotImplementedError:
            cpus = 2  # Default to 2 CPUs if count not available

        with mp.Pool(cpus) as pool:
            combinations = product(
                range(*window_range),
                range(*days_high_range),
                range(*stoploss_range),
            )
            results = pool.starmap(self.optimize, combinations)

        if results:
            df_optimized = pd.concat(results)
            if not df_optimized.empty:
                df_optimized.sort_values(by="strat_ret", ascending=False, inplace=True)
                return df_optimized
        return pd.DataFrame()