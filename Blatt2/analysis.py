import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from scipy.optimize import differential_evolution

# -------------------------------------------------
# Sinus
# -------------------------------------------------

def sine_function(t, amplitude, frequency, phase):
    w = 2*np.pi*frequency
    return - amplitude * w**2 * np.sin(w*t + phase)

# -------------------------------------------------
# Daten
# -------------------------------------------------

times = []
accel_abs = []

start = 150
count = 400

with open('sensordata.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)

    for i, row in enumerate(csv_reader):
        if i < start:
            continue
        if i >= start + count:
            break

        times.append(float(row[0].replace(',', '.')))
        accel_abs.append(float(row[4].replace(',', '.')) - 9.81)

times = np.array(times) - times[0]
accel_abs = np.array(accel_abs)

# -------------------------------------------------
# Fitness
# -------------------------------------------------

def fitness(params):
    amplitude, frequency, phase = params
    pred = sine_function(times, amplitude, frequency, phase)
    return np.sum((accel_abs - pred)**2)

# -------------------------------------------------
# Bounds
# -------------------------------------------------

bounds = [
    (0, 50),
    (0.1, 20),
    (-np.pi, np.pi)
]

# -------------------------------------------------
# Configs
# -------------------------------------------------

configs = [
    ("best1bin", 20, 0.5, 0.7, 100),
    ("best1bin", 50, 0.5, 0.7, 100),
    ("rand1bin", 20, 0.5, 0.7, 100),
    ("best2bin", 20, 0.9, 0.9, 100),
]

runs = 10

# -------------------------------------------------
# Ergebnisse speichern
# -------------------------------------------------

labels = []
mean_errors = []
mean_iters = []
mean_runtimes = []

# -------------------------------------------------
# Plot Sinus
# -------------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(times, accel_abs, color='red', s=10, label='Sensor Data')

# -------------------------------------------------
# DE + Mittelwerte
# -------------------------------------------------

for strategy, popsize, mutation, recombination, maxiter in configs:

    errors = []
    iters = []
    runtimes = []
    results_store = []

    for _ in range(runs):

        start_time = time.time()

        result = differential_evolution(
            fitness,
            bounds,
            strategy=strategy,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            maxiter=maxiter,
            disp=False
        )

        end_time = time.time()

        errors.append(result.fun)
        iters.append(result.nit)
        runtimes.append(end_time - start_time)

        results_store.append(result)

    # Mittelwerte
    mean_error = np.mean(errors)
    mean_iter = np.mean(iters)
    mean_runtime = np.mean(runtimes)

    mean_errors.append(mean_error)
    mean_iters.append(mean_iter)
    mean_runtimes.append(mean_runtime)

    labels.append(f"{strategy} pop={popsize}")

    # bester Run für Plot
    best_result = results_store[np.argmin(errors)]

    amp, freq, phase = best_result.x

    y = sine_function(times, amp, freq, phase)

    plt.plot(
        times,
        y,
        label=f"{strategy} err={mean_error:.2f}"
    )

    print(
        strategy,
        "Mean Error:", mean_error,
        "Mean Iter:", mean_iter,
        "Mean Runtime:", mean_runtime
    )

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Sinus Approximation (Mean over runs)")
plt.legend()
plt.show()