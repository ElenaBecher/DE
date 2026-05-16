import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import differential_evolution

#neue sinusfunktion: Aw^2*sin(wt+p)
def sine_function(t, amplitude, frequency, phase):
    w = 2*np.pi*frequency
    return - amplitude * w**2 * np.sin(w*t +phase)

# Read data from CSV file (only first 200 lines)
times = []
accel_abs = []

# with open('sensordata.csv', 'r') as file:
#     csv_reader = csv.reader(file, delimiter=';')
#     next(csv_reader)  # Skip header row
#     for i, row in enumerate(csv_reader):
#         if i >= 1000:
#             break
#         times.append(float(row[0].replace(',', '.')))
#         accel_abs.append(float(row[4].replace(',', '.')) - 9.81)


# die ersten haben eine weswentlich klienere amplitude, diese überspringen

start = 150
count = 400

with open('sensordata.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)  # Header überspringen

    for i, row in enumerate(csv_reader):
        if i < start:
            continue
        if i >= start + count:
            break

        times.append(float(row[0].replace(',', '.')))
        accel_abs.append(float(row[4].replace(',', '.')) - 9.81)



#Normalize time values
times = np.array(times) - times[0]
accel_abs = accel_abs

# scrpy übergibt array, daher zwischenfunktion bnem aufruf von sin
def fitness(params):
    amplitude, frequency, phase = params
    prediction = sine_function(times, amplitude, frequency, phase)
    # gesamtdeher. Abstand zw Messung und FUnktion. Quadriert, wegen negativen zahlen
    # Abstand jeder Messwert - Funktion => muss minimal werden
    errorRate = np.sum((accel_abs - prediction)**2)
    return errorRate


#DE anwenden


strategy = input("Strategy (best1bin, best1exp,rand1bin, best2bin,... ): ")
popsize = int(input("Populationsgröße: "))
mutation = float(input("Scale Factor F: "))
recombination = float(input("Crossover CR: "))
maxiter = int(input("Max Iterationen: "))

#i just fucked aroung and choose them.
#probably way to high. or low.
bounds = [
    (0, 50),        # amplitude
    (0.1, 20),      # frequency
    (-np.pi, np.pi)  # phase
]

result = differential_evolution(
    fitness,
    bounds,
    strategy=strategy,
    popsize=popsize,
    mutation=mutation,
    recombination=recombination,
    maxiter=maxiter,
    disp=True
)

print(result.x)

# Plot sine function and sensor data points
amplitude_opt, frequency_opt, phase_opt = result.x
y_sine = sine_function(times, amplitude_opt, frequency_opt, phase_opt)

plt.plot(times, y_sine, label='Sine Function')
plt.scatter(times, accel_abs, color='red', label='Sensor Data')

plt.xlabel('Time (normalized)')
plt.ylabel('Amplitude')
plt.title('Sine Function and Sensor Data')
plt.legend()
plt.show()
