from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
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
    
    scaler = preprocessing.StandardScaler()
    RSSI_train_scaled = scaler.fit_transform(RSSI_train)
    RSSI_test_scaled = scaler.transform(RSSI_test)
    # Create a Gaussian process regressor with an RBF kernel
    kernel = RBF(1, (1e-100, 1e100)) # Research states the kind of kernel doesn't really matter
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit the Gaussian process regressor to data
    gp.fit(RSSI_train_scaled, loc_train)

    # Predict using the trained Gaussian process regressor
    y_pred = gp.predict(RSSI_test_scaled)

    # Return the predicted x,y values
    return y_pred

def gaussian_gen_data(loc_train, RSSI_train, n_predictions):
    """
    Create a Gaussian process regressor with an RBF kernel, fit it to the data, and generate training data.

    Parameters:
    loc_train (array-like): The input data to fit the Gaussian Process Regressor.
    RSSI_train (array-like): The target values corresponding to the input data.
    n_predictions (int): The multiplier per RSSI value provided.

    Returns:
    y_pred (array-like): The generated RSSI values.
    """
    
    #scaler = preprocessing.StandardScaler()
    #RSSI_train_scaled = scaler.fit_transform(RSSI_train)
    #RSSI_test_scaled = scaler.transform(RSSI_test)
    
    # Create a Gaussian process regressor with an RBF kernel
    kernel = RBF(length_scale=6.63, length_scale_bounds=(1e-10, 1e10)) # Research states the kind of kernel doesn't really matter
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1)

    # Fit the Gaussian process regressor to data
    gp.fit(loc_train, RSSI_train)
    
    print("KERN")
    print(gp.kernel_)

    # Predict using the trained Gaussian process regressor
    x = list(np.arange(3, 9, 0.25))
    y = list(np.arange(1, 7, 0.25))
    #print(x, y)
    X_test = np.array([[xi, yi] for xi, yi in zip(x, y)])
    y_pred = gp.sample_y(X_test, n_predictions, random_state=42)

    # Return the predicted RSSI values
    return x, y, y_pred


if __name__ == "__main__":
    LOC_train = np.array([[1.1], [2.2], [3.3]])
    RSSI_train = np.array([[-80, -65, -70], [-55, -60, -15], [-50, -25, -80]])
    RSSI_test = np.array([[-58, -63, -68], [-53, -58, -73]])
    x = LOC_train  # Your input data
    y = RSSI_train  # Your target values
    print(gaussian_gen_data(x, y, n_predictions=3))  # Make predictions
