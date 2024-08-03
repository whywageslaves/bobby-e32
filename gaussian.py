from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def gaussian(x, y, n_predictions):
    """
    Fit a Gaussian Process Regressor to the provided data and make predictions.

    This function creates a Gaussian Process Regressor using an RBF kernel, fits it to the input data `x` (location ID's) and target values `y` (RSSI values), 
    and then returns a prediction of RSSI values.

    Parameters:
    x (array-like): The input data to fit the Gaussian Process Regressor. (i.e location ID or x,y coords)
    y (array-like): The target values corresponding to the input data. (i.e RSSI values)
    n_predicitons (int): The number of predictions to make per RSSI value provided.

    Returns:
    y_pred (array-like): The predicted values for the test input data.
    """
    # Create a Gaussian process regressor with an RBF kernel
    kernel = RBF() # Research states the kind of kernel doesn't really matter
    gp = GaussianProcessRegressor(kernel=kernel)

    # Fit the Gaussian process regressor to data
    gp.fit(x, y)

    # Predict using the trained Gaussian process regressor
    X_test = np.array(list(range(1, len(y)+1))).reshape(-1, 1)
    y_pred = gp.sample_y(X_test, n_predictions, random_state=42)

    # Return the predicted RSSI values
    return y_pred


if __name__ == "__main__":
    LOC_train = np.array([[1], [2], [3]])
    RSSI_train = np.array([[-80, -65, -70], [-55, -60, -15], [-50, -25, -80]])
    RSSI_test = np.array([[-58, -63, -68], [-53, -58, -73]])
    x = LOC_train  # Your input data
    y = RSSI_train  # Your target values
    print(gaussian(x, y, n_predictions=3))  # Make predictions
