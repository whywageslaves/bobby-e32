import matplotlib.pyplot as plt

# Example data
full_data_errors = [0.78, 1.82, 0.54, 1.53, 1.94, 1.58, 0.92, 0.67, 0.73, 0.80, 0.22, 0.85, 2.62, 2.31]
half_data_errors = [2.26, 0.50, 1.94, 1.53, 1.94, 0.54, 1.94, 0.67, 0.73, 0.80, 1.80, 0.85, 2.62, 2.31]
llm_data_errors = [2.26, 0.50, 1.94, 1.53, 1.94, 0.54, 1.94, 0.67, 0.73, 0.80, 1.80, 0.85, 2.62, 2.31]

data = [full_data_errors, half_data_errors, llm_data_errors]
labels = ['Full Data', 'Half Data', 'LLM Data']

# Creating the box plot
plt.boxplot(data, labels=labels)
plt.ylabel('Distance Errors (m)')
plt.savefig("plot.png")
