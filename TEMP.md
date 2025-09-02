

# pytrendy: A Trend Detection and Analysis Library for Time-Series Data

pytrendy is a Python package designed for the automated detection, analysis, and visualization of trends in time-series data. It provides a robust and customizable pipeline that identifies periods of upward, downward, flat, and noisy behavior, enabling data analysts and scientists to quickly and accurately find meaningful changes in their data. The library is built to be resilient to common time-series challenges like seasonality and noise, offering refined results and insightful metrics.

## 1. The detect_trends() Function and pytrendyResults Class

The detect_trends() function is the central orchestrator of the PyTrendy analysis pipeline. It takes a DataFrame, processes the data through a series of internal steps, and returns a PyTrendyResults object.

The function's internal pipeline follows a specific order:

- process_signals(): Smooths the data and identifies trend directions, flat periods, and noise.
- get_segments(): Groups consecutive data points with the same trend direction into segments.
- refine_segments(): Adjusts segment boundaries and classifies trends as gradual or abrupt.
- analyse_segments(): Calculates key metrics and ranks the trends by their significance.
- plot_pytrendy(): Generates the plot if plot=True.

The final output is the PyTrendyResults object.


### The PyTrendyResults Class

This class is a wrapper for the analysis results. It provides a structured way to access and interact with the detected segments.

Key Attributes:

- self.segments: A list of dictionaries, where each dictionary represents a single detected segment and its associated metrics.
- self.best: The most significant trend found, determined by its total_change (the cumulative sum of differences over the segment). This metric prioritizes both the duration and  magnitude of the change.
- self.summary: A dictionary containing high-level statistics, such as a count of each trend type (Up, Down, Flat, Noise).
- self.segments_df: A pandas.DataFrame representation of the segments, which is often more convenient for data manipulation and programmatic access than the list of dictionaries.

## 2. PyTrendy's Signal Processing: The process_signals() Function

The process_signals() function is the first and most critical step in the PyTrendy pipeline. It prepares your raw time-series data for trend detection by applying several statistical methods.


Process Overview:

This function works by creating new columns on your DataFrame that flag each data point based on its characteristics (upward trend, downward trend, flat, or noisy).

Data Smoothing: The function begins by smoothing the data using a Savitzky-Golay filter. This is more advanced than a simple rolling average; it reduces noise while preserving the shape and peaks of the original data. The WINDOW_SMOOTH parameter controls the level of smoothing.

Flatness Detection: It identifies periods of little to no change by calculating the rolling standard deviation of the smoothed data. If the standard deviation within a rolling window falls below a THRESHOLD_FLAT, that period is flagged as "flat."

Noise Detection: To distinguish a true trend from random fluctuations, the function calculates the Signal-to-Noise Ratio (SNR) for each data point. It flags periods with a low SNR (below THRESHOLD_NOISE) as "noisy." This ensures that the final trend detection is not overly sensitive to erratic data points.

Trend Detection: The main trend flags (uptrend or downtrend) are assigned using the first derivative of the smoothed data. A positive derivative indicates an upward trend, while a negative one indicates a downward trend. A THRESHOLD_SMOOTH parameter is used to ensure only significant changes are flagged as trends, and any periods previously flagged as flat or noisy are excluded from being classified as a trend.

The function returns the modified DataFrame with all the new flags, ready for the next step in the pipeline.



## 2. Analysis and Reporting : segments_analyse.py

This module enriches each segment with quantitative metrics, providing a deeper understanding of its characteristics.

### Change Metrics
Calculates the absolute and percentage change from the start to the end of the trend. For up/down trends, it also computes the total_change, which is the cumulative sum of daily differences, providing a robust measure of the trend's magnitude.

### Signal-to-Noise Ratio (SNR)
Computes the SNR for the segment, which indicates the strength of the trend relative to the noise.

## 3. Output and Visualization of pytrendy

The plot_pytrendy.py module creates a clear visualization of the trend analysis. It takes your time-series data and overlays shaded regions to highlight different trend segments. 

- Uptrends : Green
- Downtrends : Red
- Flat : Blue
- Noise : Gray

The trends are further labeled with their rank according to their impact to quickly identify the biggest changes in the data.

## 4. The detect_trends() Function of pytrendy

The detect_trends() function is the primary entry point for the pytrendy library. You simply provide it with your data and specify the columns you want to analyze.

- df: Your data as a pandas DataFrame.
- date_col: The name of the column that contains the dates.
- value_col: The name of the column with the numerical values to be analyzed.
- plot: A boolean flag. Set it to True (the default) to generate the plot, or False to skip the visualization.

This function runs the entire trend detection process and returns a pytrendyResults object.

### The pytrendyResults Object
The pytrendyResults object acts as a container for all the analysis output. It provides a structured way to access the results through various attributes and methods.

- segments: A list of dictionaries, where each dictionary holds the details and calculated metrics for a single trend segment.
- best: The single most significant trend found in your data, determined by its change rank.
- summary: A high-level overview of the analysis, including the number of each type of trend (e.g., how many uptrends were found).
- print_summary(): A method that displays a formatted, easy-to-read summary of the results in your console.
- segments_df: The full trend results presented as a pandas DataFrame, which is useful for data manipulation.
- filter_segments(): A powerful method to query the results. You can use it to filter segments by direction (Up, Down, etc.) and sort them by rank or time.