#..........................................................Part1a_1190298 & 1180546 & 1190747..........................................................#

import matplotlib.pyplot as plt
import numpy as np

sample_count = 2000
sample_index = np.arange(sample_count)
input_signal = np.cos(0.03*np.pi*sample_index)
plt.figure("Part1a_1190298 & 1180546 & 1190747")
plt.plot(sample_index, input_signal)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Input Signal: x[n] = cos(0.03πn)')
plt.show()

#..........................................................Part1b_1190298 & 1180546 & 1190747..........................................................#

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

N = 2000

transfer_function_coefficients = np.array([1, -2, 4])

w, h = signal.freqz(transfer_function_coefficients, 1, N)


plt.figure("Part1b_1190298 & 1180546 & 1190747")

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('dB')
plt.title('Amplitude response')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(h))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('Radians')
plt.title('Phase response')
plt.subplots_adjust(hspace=0.5)
plt.grid()

plt.show()

#..........................................................Part1c_1190298 & 1180546 & 1190747..........................................................#

import numpy as np
import matplotlib.pyplot as plt

num_samples = 2000
indices = np.arange(num_samples)
input_signal = np.cos(0.03 * np.pi * indices)

spectrum = np.fft.fft(input_signal)
frequencies = np.linspace(-num_samples / 2, num_samples / 2, num_samples)

plt.figure("Part1b_1190298 & 1180546 & 1190747")
plt.plot(frequencies, 20 * np.log10(np.abs(np.fft.fftshift(spectrum))))
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.title("Symmetric Spectrum for Input Signal")
plt.show()

#..........................................................Part1d_1190298 & 1180546 & 1190747..........................................................#
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

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x[n], x[n-1], x[n-2], x[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1d_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1e_1190298 & 1180546 & 1190747..........................................................#
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

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x[n], x[n-1], x[n-2], x[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the amplitude and phase response for the estimated FIR system at the end of the iterations.
w, h = signal.freqz(filterCo, 1, N)

plt.figure("Part1e_1190298 & 1180546 & 1190747")

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('dB')
plt.title('Amplitude response')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(h))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('Radians')
plt.title('Phase response')
plt.subplots_adjust(hspace=0.5)
plt.grid()

plt.show()

#..........................................................Part1f_1190298 & 1180546 & 1190747..........................................................#
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
stepSize = 0.001

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x[n], x[n-1], x[n-2], x[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1f_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1g_d_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Add 40dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
x_noisy = x + noise


# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1g_d_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1g_e_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)


# Add 40dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
x_noisy = x + noise

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the amplitude and phase response for the estimated FIR system at the end of the iterations.
w, h = signal.freqz(filterCo, 1, N)

plt.figure("Part1g_e_1190298 & 1180546 & 1190747")

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('dB')
plt.title('Amplitude response')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(h))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('Radians')
plt.title('Phase response')
plt.subplots_adjust(hspace=0.5)
plt.grid()

plt.show()

#..........................................................Part1g_f_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Add 40dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-40/10)), N)
x_noisy = x + noise

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.001

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1g_f_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1h_d_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Add 30dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-30/10)), N)
x_noisy = x + noise


# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1h_d_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1h_e_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)


# Add 30dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-30/10)), N)
x_noisy = x + noise

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x_noisy) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.01

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the amplitude and phase response for the estimated FIR system at the end of the iterations.
w, h = signal.freqz(filterCo, 1, N)

plt.figure("Part1h_e_1190298 & 1180546 & 1190747")

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('dB')
plt.title('Amplitude response')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(h))
plt.xlabel('Frequency (pi radians/sample)')
plt.ylabel('Radians')
plt.title('Phase response')
plt.subplots_adjust(hspace=0.5)
plt.grid()

plt.show()

#..........................................................Part1h_f_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt

# Define the length of the input signal
N = 2000

# Generate the input signal x[n] = cos(0.03*pi*n)
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

# Add 30dB of zeros mean white Gaussian noise to x[n]
noise = np.random.normal(0, np.sqrt(10**(-30/10)), N)
x_noisy = x + noise

# Define the unknown system and the adaptive filter
sysCo = [1, -2, 4] # Unknown system coefficients
adaptiveCo = [1] # Adaptive filter coefficients
desired = np.convolve(sysCo, x) # Output of the unknown system
desired = desired[:N]

# Define the initial filter coefficients
filterCo = np.zeros(4)

# Define the step size factor
stepSize = 0.001

# Initialize variables for the LMS algorithm
adptive_out = np.zeros(N)
error = np.zeros(N)
J = np.zeros(N)

# Implement the LMS algorithm
for n in range(3, N-1):
    xn_input = np.array([x_noisy[n], x_noisy[n-1], x_noisy[n-2], x_noisy[n-3]]) # Current input sample
    adptive_out[n] = np.dot(xn_input, filterCo) # Output of the adaptive filter
    error[n] = desired[n] - adptive_out[n] # Error between the output of the unknown system and the adaptive filter
    filterCo = filterCo + 2 * stepSize * error[n] * xn_input # Update the filter coefficients
    J[n] = error[n]**2 # Compute J = e^2(n)

# Plot the estimated filter coefficients
plt.figure("Part1h_f_1190298 & 1180546 & 1190747")

plt.subplot(2,2,1)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.stem(np.arange(4), filterCo,markerfmt='k', linefmt='k')
plt.xlabel('Index')
plt.title('New Filter Coefficients')

# Plot the error e(n)
plt.subplot(2,2,2)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(error,color='k')
plt.xlabel('Index')
plt.title('Error')

# Plot J versus iteration steps
plt.subplot(2,2,3)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(J,color='k')
plt.xlabel('Index')
plt.title('J vs. Index')

# Plot 10log10(J) versus iteration steps
plt.subplot(2,2,4)
plt.axhline(y=0, color='m')
plt.axvline(x=0, color='m')
plt.plot(10 * np.log10(J),color='k')
plt.xlabel('Index')
plt.ylabel('dB')
plt.title('10log10(J) vs. Index')
plt.subplots_adjust(hspace=0.5)
plt.show()

#..........................................................Part1i_1190298 & 1180546 & 1190747..........................................................#
import numpy as np
import matplotlib.pyplot as plt

N = 2000 # number of samples
x = np.cos(0.03*np.pi*np.arange(N)) # input signal

# Add 40dB of zero mean white Gaussian noise to x[n]
x = x + np.random.normal(0, 1, N)*10**(-40/10)

trials = 1000 # number of trials
J_avg = np.zeros(N) # initialize average cost function J

for t in range(trials):
    adaptiveCo = np.zeros(4) # initialize filter coefficients
    error = np.zeros(N) # initialize error
    stepSize = 0.01 # step size factor
    for n in range(4, N):
        xn = np.array([x[n], x[n-1], x[n-2], x[n-3]]) # input vector
        y = np.dot(adaptiveCo, xn) # filter output
        desired = 1 - 2*x[n-1] + 4*x[n-2] # desired output
        error[n] = desired - y # error
        adaptiveCo = adaptiveCo + 2 * stepSize * error[n] * xn # update filter coefficients using LMS
        J_avg[n] = J_avg[n] + error[n]**2 # update cost function

J_avg = J_avg/trials # average cost function over number of trials

plt.figure("Part1i_1190298 & 1180546 & 1190747")

plt.plot(10*np.log10(J_avg)) # plot averaged J (10log10(J) vs iteration steps)
plt.xlabel('Iteration steps')
plt.ylabel('10log10(J)')
plt.title('Averaged J (10log10(J) vs iteration steps) using LMS')
plt.show()
