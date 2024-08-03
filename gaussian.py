from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def gaussian_predict_location(RSSI_train, loc_train, RSSI_test):
    """
    Fit a Gaussian Process Regressor to the provided data and make predictions.

    This function creates a Gaussian Process Regressor using an RBF kernel, fits it to the input data `RSSI_train` (RSSI values) and target values `loc_train` (location ID's), 
    and then returns a prediction of location based on RSSI_test (validation data).

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
    gp.fit(RSSI_train, loc_train)

    # Predict using the trained Gaussian process regressor
    y_pred = gp.predict(RSSI_test)

    # Return the predicted x,y values
    return y_pred

def gaussian_gen_data(x, y, n_predictions):
    """
    Create a Gaussian process regressor with an RBF kernel, fit it to the data, and generate training data.

    Parameters:
    x (array-like): The input data to fit the Gaussian Process Regressor.
    y (array-like): The target values corresponding to the input data.
    n_predictions (int): The multiplier per RSSI value provided.

    Returns:
    y_pred (array-like): The generated RSSI values.
    """
    
    # Create a Gaussian process regressor with an RBF kernel
    kernel = RBF() # Research states the kind of kernel doesn't really matter
    gp = GaussianProcessRegressor(kernel=kernel)

    # Fit the Gaussian process regressor to data
    gp.fit(x, y)

    # Predict using the trained Gaussian process regressor
    X_test = np.array(list(range(1, len(y)+1))).reshape(-1, 1)
    y_pred = gp.sample_y(X_test, n_predictions, random_state=42)
    y_pred2 = gp.predict(X_test)

    # Return the predicted RSSI values
    return y_pred2


if __name__ == "__main__":
    LOC_train = np.array([[1.1], [2.2], [3.3]])
    RSSI_train = np.array([[-80, -65, -70], [-55, -60, -15], [-50, -25, -80]])
    RSSI_test = np.array([[-58, -63, -68], [-53, -58, -73]])
    x = LOC_train  # Your input data
    y = RSSI_train  # Your target values
    print(gaussian_gen_data(x, y, n_predictions=3))  # Make predictions
