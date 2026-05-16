import numpy as np
import matplotlib.pyplot as plt
import csv

# Define sine function with variable parameters
#def sine_function(t, amplitude, frequency, phase, displacement):
#    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + displacement

#neue sinusfunktion: Aw^2*sin(wt+p)
def sine_function(t, amplitude, frequency, phase):
    w = 2*np.pi*frequency
    return - amplitude * w**2 * np.sin(w*t +phase)

# Read data from CSV file (only first 200 lines)
times = []
accel_abs = []
with open('sensordata.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)  # Skip header row
    for i, row in enumerate(csv_reader):
        if i >= 1000:
            break
        times.append(float(row[0].replace(',', '.')))
        accel_abs.append(float(row[4].replace(',', '.')) - 9.81)

# Normalize time values
times = np.array(times) - times[0]
accel_abs = accel_abs

# werte, die wir optimieren wollen
amplitude = 15
frequency = 0.101
phase = 1.1


# Generate y values for the sine function using normalized time values
#y_sine = sine_function(times, amplitude, frequency, phase, displacement)
y_sine = sine_function(times, amplitude, frequency, phase)

# Plot sine function and sensor data points
plt.plot(times, y_sine, label='Sine Function')
plt.scatter(times, accel_abs, color='red', label='Sensor Data')

plt.xlabel('Time (normalized)')
plt.ylabel('Amplitude')
plt.title('Sine Function and Sensor Data (First 200 Lines)')
plt.legend()
plt.show()
