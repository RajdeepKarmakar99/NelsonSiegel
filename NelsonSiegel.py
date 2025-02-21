import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Nelson-Siegel function calculates the yield for a given maturity 't'
def NelsonSiegel(b0, b1, b2, tau, t):
    """
    Computes the yield using the Nelson-Siegel model.

    Parameters:
        b0, b1, b2: Model parameters that capture level, slope, and curvature.
        tau: Decay factor (must be positive to avoid division by zero).
        t: Maturity (in years).

    Returns:
        Calculated yield for the given maturity.
    """
    # Term A helps capture the short-term behavior
    A = (1 - np.exp(-t/tau)) / (t/tau)
    # Term B adjusts for the curvature (medium-term behavior)
    B = A - np.exp(-t/tau)
    return b0 + b1 * A + b2 * B

# Optimizer function calibrates the model to observed yields
def Optimizer(observed_yields, maturities):
    """
    Finds the best-fitting parameters for the Nelson-Siegel model
    by minimizing the sum of squared errors between actual and model yields.
    
    Parameters:
        observed_yields: Array of yields from market data.
        maturities: Array of maturities corresponding to the yields.
        
    Returns:
        Optimal parameters: b0, b1, b2, and tau.
    """
    # Define the objective function to minimize
    def objective(params, actual, t):
        # params contains [b0, b1, b2, tau]
        model_yields = NelsonSiegel(params[0], params[1], params[2], params[3], t)
        # Sum of squared differences between actual and modeled yields
        return np.sum((actual - model_yields) ** 2)

    # Initial guess for the parameters: b0, b1, b2, tau
    initial_guess = [0.1, 0.1, 0.1, 0.1]
    # Set bounds; tau must be positive to avoid division by zero.
    parameter_bounds = [(None, None), (None, None), (None, None), (0.001, None)]
    # Use the minimize function from SciPy to calibrate the model
    result = minimize(objective, initial_guess, args=(observed_yields, maturities), bounds=parameter_bounds)
    return result.x

# ---------------------------
# Data Loading and Preparation
# ---------------------------
# Load yield data from an Excel file.
data = pd.read_excel('InterestRate.xlsx')

# Extract the 'Date' column and yield columns (assumes first column is 'Date')
dates = data['Date'].values
yield_columns = data.columns.tolist()
yield_columns.remove('Date')  # Remove 'Date' from the list

# Convert the yield data from percentages to decimals
yield_values = data[yield_columns].values / 100.0

# Define the maturities (in years). Here, maturities are given in months and converted to years.
maturities = np.array([1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]) / 12.0

# ---------------------------
# Model Calibration and Forecasting
# ---------------------------
# Use the most recent row of data for calibration
current_yields = yield_values[-1]
# Calibrate the model parameters using the optimizer
b0, b1, b2, tau = Optimizer(current_yields, maturities)
# Compute the fitted yields using the calibrated parameters
fitted_yields = NelsonSiegel(b0, b1, b2, tau, maturities)

# ---------------------------
# Plotting the Results
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(maturities, current_yields, 'ro-', label='Actual Yields')
plt.plot(maturities, fitted_yields, 'bo-', label='Fitted Nelson-Siegel Curve')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (Decimal)')
plt.title('Nelson-Siegel Yield Curve Fit')
plt.legend()
plt.show()
