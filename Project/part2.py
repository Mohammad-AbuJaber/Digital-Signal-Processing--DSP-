#......................................................................Part2_D.......................................

import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [x[n], x[n-1], x[n-2], x[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_d 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_E.......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [x[n], x[n-1], x[n-2], x[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

#Plot the amplitude and phase response for the estimated FIR system
adptive_out, Freq_h = signal.freqz(adptive_out, 1, N)

plt.figure("Part2_E 1180546 & 1190298 & 1190747   : NLMS")


plt.subplot(2, 1, 1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, 20 * np.log10(np.abs(Freq_h)) ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('DB')
plt.title('Amplitude : 1180546 & 1190298 & 1190747')

plt.subplot(2, 1, 2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, np.unwrap(np.angle(Freq_h)) / np.pi ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('Phase')
plt.title('Phase : 1180546 & 1190298 & 1190747')

plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_F.......................................


import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Decrease the step size factor
#stepSize = 0.01
stepSize = 0.001

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [x[n], x[n-1], x[n-2], x[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_F 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_G_D.......................................


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

#Add 40dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 40
#generating the  white Gaussian noise with specific 40 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

#step size factor
stepSize = 0.01


# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_G_D 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()



#......................................................................Part2_G_E.......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import randn


# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

#Add 40dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 40
#generating the  white Gaussian noise with specific 40 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

#Plot the amplitude and phase response for the estimated FIR system
adptive_out, Freq_h = signal.freqz(adptive_out, 1, N)

plt.figure("Part2_G_E 1180546 & 1190298 & 1190747   : NLMS")


plt.subplot(2, 1, 1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, 20 * np.log10(np.abs(Freq_h)) ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('DB')
plt.title('Amplitude : 1180546 & 1190298 & 1190747')

plt.subplot(2, 1, 2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, np.unwrap(np.angle(Freq_h)) / np.pi ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('Phase')
plt.title('Phase : 1180546 & 1190298 & 1190747')

plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_G_F.......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)


#Add 40dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 40
#generating the  white Gaussian noise with specific 40 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 


# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Decrease the step size factor
#stepSize = 0.01
stepSize = 0.001

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_G_F 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()



#......................................................................Part2_H_D.......................................



import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

#Add 30dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 30
#generating the  white Gaussian noise with specific 30 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

#step size factor
stepSize = 0.01


# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_H_D 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_H_E.......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import randn


# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

#Add 30dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 30
#generating the  white Gaussian noise with specific 30 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

#Plot the amplitude and phase response for the estimated FIR system
adptive_out, Freq_h = signal.freqz(adptive_out, 1, N)

plt.figure("Part2_H_E 1180546 & 1190298 & 1190747   : NLMS")


plt.subplot(2, 1, 1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, 20 * np.log10(np.abs(Freq_h)) ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('DB')
plt.title('Amplitude : 1180546 & 1190298 & 1190747')

plt.subplot(2, 1, 2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(adptive_out / np.pi, np.unwrap(np.angle(Freq_h)) / np.pi ,color='k')
plt.xlabel('N_Frequency')
plt.ylabel('Phase')
plt.title('Phase : 1180546 & 1190298 & 1190747')

plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_H_F_MU1 = 0.001 .......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)


#Add 30dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 30
#generating the  white Gaussian noise with specific 30 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 


# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Decrease the step size factor
#stepSize = 0.01
stepSize = 0.001

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_H_F_MU1=0.001 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()

#......................................................................Part2_H_F_MU2 = 0.1 .......................................

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)


#Add 30dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 30
#generating the  white Gaussian noise with specific 30 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 


# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Increase the step size factor
#stepSize = 0.01
stepSize = 0.1

# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the NLMS algorithm
for n in range(3, N):
    xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part2_H_F_MU2=0.1 1180546 & 1190298 & 1190747   : NLMS")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients : 1180546 & 1190298 & 1190747')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error : 1180546 & 1190298 & 1190747')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index : 1180546 & 1190298 & 1190747')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index : 1180546 & 1190298 & 1190747')
plt.subplots_adjust(hspace=0.5)
plt.show()


#......................................................................Part2_I .......................................

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# Define the length of the input signal
N = 2000

# Generate the input signal x[n]
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

#Add 40dB of zeros mean white Gaussian noise to  the input signal 
SignalToNoiseRatio = 40
#generating the  white Gaussian noise with specific 40 db SignalToNoiseRatio
noise = randn(N) / np.power(10, (SignalToNoiseRatio / 20)) 
#Adding noise to the input signal
input_With_noisy = x + noise 

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, input_With_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

#step size factor
stepSize = 0.01


# Initialize variables for the NLMS algorithm
adptive_out = np.zeros(N)

# number of trials
trials = 1000 
J_avg = np.zeros(N) # initialize average cost function J

for t in range(trials):
    filterCo = np.zeros(4) # initialize filter coefficients
    error = np.zeros(N) # initialize error
    # Implement the NLMS algorithm
    for n in range(3, N):
        xn_input = [input_With_noisy[n], input_With_noisy[n-1], input_With_noisy[n-2], input_With_noisy[n-3]] # Current input sample
        adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
        error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
        filterCo = filterCo + stepSize * error[n] * np.array(xn_input) / (np.dot(xn_input, xn_input) + 1e-6)  # Update the filter coefficients
        J_avg[n] = J_avg[n] + error[n]**2 # update cost function

# average cost function over number of trials : "1000" 
J_avg = J_avg / trials 

plt.figure("Part2_I 1180546 & 1190298 & 1190747   : NLMS")
# plot averaged J (10log10(J) vs iteration steps)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10*np.log10(J_avg),color='k')
plt.xlabel('Index')
plt.ylabel('10log10(Average_J)')
plt.title('Part2_I 1180546 & 1190298 & 1190747   : NLMS')
plt.show()
