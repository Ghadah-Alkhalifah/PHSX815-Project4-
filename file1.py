import numpy as np
from scipy.optimize import curve_fit
from numba import jit # For optimization
import random

# h=0 no magnetic field
X=10 #Size of the lattice
Y=10 # Size of the lattice
# Set up the lattice
seed=5555
random.seed(seed)
lattice = np.random.choice([-1, 1], size=(X, Y)) #initial state

# Define the neighbor spins function
@jit(nopython=True, cache=True)
def Val_neighbors(L): #L means the lattice
    N, M = L.shape
    NB = np.zeros_like(L)
    for i in range(X):
        for j in range(Y):
            xi, xj = (i+1)%X, (j+1)%Y #set boundry condition depending on the site of spin
            xi2, xj2=(i-1)%X, (j-1)%Y
            NB[i, j]  = L[xi, j] + L[xi2, j] + L[i, xj] + L[i, xj2]
        return NB

#Define the energy function
@jit(nopython=True, cache=True)
def E_Val(L):
    E = 0
    NB = Val_neighbors(L)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            s = L[i, j]
            E += -s * NB[i, j]
    E /= 4.0
    return E

# Define the Monte Carlo update function
@jit(nopython=True, cache=True)
def montC(L, T):
    # Define a function to calculate the energy change for flipping a spin at (i, j)
    def dE(i, j):
        s = L[i, j]
        xi, xj = (i+1)%X, (j+1)%Y
        xi2, xj2=(i-1)%X, (j-1)%Y
        NB = L[xi, j] + L[xi2, j] + L[i, xj] + L[i, xj2]
        return 2 * s * NB

    # Loop over all lattice sites
    for i in range(X):
        for j in range(Y):
            # Calculate the energy change for flipping the spin at (i, j)
            dE_Val = dE(i, j)
            # If flipping the spin would lower the energy or the flip is accepted
            if dE_Val < 0 or np.exp(-dE_Val/T) > np.random.rand():
                # Flip the spin at (i, j)
                L[i, j] *= -1

    # Return the updated lattice
    return L



def quantity(X, Y, T_min, T_max, T_step, MCsweep, Nbootstrap):
    # Arrays to store quantities data
    E = []
    S_heat = []
    temperatures = np.arange(T_min, T_max, T_step)

    # Run the Monte Carlo simulation for each temperature
    for T in temperatures:
        L = np.random.choice([-1, 1], size=(X, Y))
        E_total = 0
        E_2_total=0
        
        for i in range(MCsweep):
            L = montC(L, T)
            E1 = E_Val(L)
            E_total += E1
            E_2_total += E1*E1

        # Calculate the specific heat and add it to the list
        Cv = ((E_2_total / MCsweep) - (E_total / MCsweep)**2) / (T**2)
        S_heat.append(Cv)
        # Add the energy to the lists
        E.append(E_total/ MCsweep)
        
    # Define the power law function
    def Plaw(x, n, m, l):
        z=abs(x - m)
        value=n *z**(-l)
        return value
    # Fit the power law function to the specific heat data
    p0 = [1, 2.268, 0.5]  # Initial guess for the parameter values
    popt, pcov = curve_fit(Plaw, temperatures, S_heat, p0=p0)

    # Extract the critical temperature from the fitted parameters
    n_opt, m_opt, l_opt = popt  # Optimal parameter values
    criticalT = m_opt  # Critical temperature is the value of 'm' parameter
    # Estimate the uncertainty in the critical temperature
    critical_temperatures = []
    for i in range(Nbootstrap):
        # Sample with replacement from the specific heats
        bootstrap_specific_heats = np.random.choice(S_heat, size=len(S_heat), replace=True)
        # Fit the power law function to the bootstrap sample
        popt, pcov = curve_fit(Plaw, temperatures, bootstrap_specific_heats, p0=p0)
        n_opt, m_opt, l_opt = popt
        critical_temperatures.append(m_opt)
    
    # Calculate the mean and standard deviation of the bootstrap samples
    mean_critical_temperature = np.mean(critical_temperatures)
    std_critical_temperature = np.std(critical_temperatures)
    
    # Print the critical temperature and its uncertainty
    print(f"The critical temperature is {criticalT:.4f} +/- {std_critical_temperature:.4f}")

    # Save the data to a file
    with open("simulation_data.txt", "w") as filex:
        for k in range(len(temperatures)):
            filex.write(f"{temperatures[k]}\t{E[k]}\t{S_heat[k]}\t{critical_temperatures[k]}\n")

    return criticalT


# parameters:
#N Size of the lattice
#M Size of the lattice
#T_min is the Minimum temperature
#T_max is the Maximum temperature
#T_step is the Temperature step
#MCsweep is the Number of Monte Carlo steps for each temperature

# Run the simulation and get the critical temperature
MC_simulation = quantity(X=10, Y=10, T_min=1.3, T_max=3.3, T_step=0.005, MCsweep = 10000, Nbootstrap=1000) 


