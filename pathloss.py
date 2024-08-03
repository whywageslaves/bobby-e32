import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class Pathloss:
    def __init__(self, distance, rssi, d0=1.0):
        """
        Initialize the Pathloss object.
        :param distance: array of distances from Tx to Rx (in meters).
        :param rssi: array of measured RSSI values (in dBm).
        :param d0: reference distance (in meters).
        """
        self.dist = distance
        self.rssi = rssi
        self.d0 = d0
        self.log_dist = np.log10(self.dist / self.d0)
        self.n = sp.symbols('n')

    def calculate_ple(self):
        """
        Calculate the path loss exponent (PLE) using the given distances and RSSI values.
        :return: the path loss exponent (n).
        """
        # Calculate the mean RSSI at reference distance d0
        rssi_d0 = np.mean(self.rssi[:np.where(self.dist == self.d0)[0][0] + 1])
        # Calculate the sum of squared errors function
        fn = sum((self.rssi - rssi_d0 + 10 * self.n * self.log_dist) ** 2)
        # Differentiate with respect to n and solve for n
        diff_fn = sp.diff(fn, self.n)
        ple = sp.solve(diff_fn, self.n)[0]
        return ple.evalf()

    def simplified_path_loss(self, ple):
        """
        Calculate the simplified path loss model using the given PLE.
        :param ple: the path loss exponent.
        :return: the simplified path loss model values (in dBm).
        """
        rssi_d0 = np.mean(self.rssi[:np.where(self.dist == self.d0)[0][0] + 1])
        pl_model = rssi_d0 - 10 * ple * self.log_dist
        return pl_model

    def plot_results(self, ple, pl_model):
        """
        Plot the measured RSSI values and the simplified path loss model.
        :param ple: the path loss exponent.
        :param pl_model: the simplified path loss model values (in dBm).
        """
        plt.plot(self.dist, self.rssi, 'o-', label='Measured RSSI')
        plt.plot(self.dist, pl_model, 's-', label=f'Simplified Path Loss Model (n={ple:.2f})')
        plt.xscale('log')
        plt.xlabel('Distance (m)')
        plt.ylabel('RSSI (dBm)')
        plt.title('Path Loss Model')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Example distances (in meters) and measured RSSI values (in dBm)
    dist = np.array([1, 10, 20, 50, 100, 200, 300])
    rssi = np.array([-40, -50, -60, -70, -80, -90, -95])
    
    # Create Pathloss object and calculate PLE
    pathloss = Pathloss(dist, rssi)
    ple = pathloss.calculate_ple()
    
    # Calculate simplified path loss model
    pl_model = pathloss.simplified_path_loss(ple)
    
    # Print PLE and plot results
    print(f'Path Loss Exponent (n): {ple}')
    pathloss.plot_results(ple, pl_model)

if __name__ == '__main__':
    main()
