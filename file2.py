import numpy as np
import matplotlib.pyplot as plt


# Load the data from the file
with open("simulation_data.txt", "r") as fileN:
    data= np.loadtxt(fileN)
    
temperatures = data[:, 0]
energies = data[:, 1]
specific_heats = data[:, 2]
criticalT=data[:, 3]

# Plot energy vs temperature
plt.figure(1)
plt.plot(temperatures, energies, '.', color='red')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.show()

# Plot specific heat vs temperature
plt.figure(2)
plt.plot(temperatures, specific_heats, '.', color='green')
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.show()


