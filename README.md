This code implements a yield curve fitting procedure using the Nelson-Siegel model. It can be summarized as follows:

Model Definition:
The NelsonSiegel function calculates yields based on parameters (b0, b1, b2) and a decay factor (tau), capturing the level, slope, and curvature of the yield curve.

Parameter Calibration:
The Optimizer function uses SciPy's minimize to calibrate these parameters by minimizing the sum of squared differences between observed market yields and those predicted by the Nelson-Siegel model.

Data Handling:
The script reads yield data from an Excel file, extracts and converts yield percentages to decimals, and defines the corresponding maturities (in years).

Model Fitting and Visualization:
Using the most recent yield data, it calibrates the model parameters, computes the fitted yields, and then plots both the actual and fitted yield curves for visual comparison.
