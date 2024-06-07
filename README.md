# BTC Pivot Points Strategy Analysis

## Overview

This project involves analyzing Bitcoin (BTC) price data to identify pivot points and develop a trading strategy based on these pivot points. The analysis is performed using historical BTC price data from Yahoo Finance, processed and visualized using Python libraries such as pandas, numpy, matplotlib, and yfinance.

## Features

- **Data Acquisition**: Fetch historical BTC-USD data from Yahoo Finance.
- **Pivot Points Calculation**: Identify pivot points based on cumulative returns over specified windows and days.
- **Trading Strategy Simulation**: Implement a trading strategy that leverages pivot points to make trading decisions, incorporating a stop-loss mechanism.
- **Strategy Optimization**: Use multiprocessing to optimize the trading strategy by testing various combinations of parameters.
- **Visualization**: Plot the cumulative returns and strategy performance, highlighting pivot points and trading signals.

## Installation

To run this project, you need to install the required Python libraries. You can install them using the following command:

## Usage

1. **Data Acquisition**: The data for BTC-USD is downloaded and processed to calculate returns and cumulative returns.
2. **Pivot Points Calculation**: The function `find_pivs` calculates pivot points based on the maximum cumulative returns over a rolling window.
3. **Trading Strategy**: The function `strat` simulates a trading strategy based on the identified pivot points and includes a stop-loss mechanism.
4. **Optimization**: The function `optimize` is used to find the best parameters (window size, days high, stoploss) for the strategy.
5. **Visualization**: Use `plot_pivots` and `plot_strat` to visualize the pivot points and the performance of the trading strategy respectively.

## Files

- `Pivot Points.ipynb`: Jupyter notebook containing all the code and documentation for the analysis.

## Running the Notebook

Ensure you have Jupyter installed, or use Google Colab to run the notebook. Open the notebook and execute the cells sequentially to reproduce the analysis.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
