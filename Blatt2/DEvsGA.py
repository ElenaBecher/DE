#!/usr/bin/python3
import random

import numpy as np
from scipy.optimize import differential_evolution


def ackley(x):
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    y = term1 + term2 + 20 + np.exp(1)
    return y


DIM = 20
BOUNDS = [(-32.768, 32.768)] * DIM
TOL = 1e-7
MAX_ITER = 10000
RUNS = 10
PRINT_EVERY = 50


def run_de(seed):
    print(f"\n** DE Run {seed} **")
    step = [0]  # Liste damit Wert geändert werden noch kann

    def callback(current_best, convergence):
        step[0] += 1
        if step[0] % PRINT_EVERY == 0:
            print(f"differential_evolution step {step[0]}: f(x) = {ackley(current_best)}")

    result = differential_evolution(
        ackley,
        BOUNDS,
        seed=seed,
        tol=TOL,
        maxiter=MAX_ITER,
        popsize=5,  # 5 * DIM = 100 Individuen
        callback=callback,  # statt disp=True
    )
    print(f"differential_evolution step {step[0]}: f(x) = {result.fun}  <-- letzter Schritt")
    return result.fun


def run_ga(seed):
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n** GA Run {seed} **")

    pop_size = 40
    pop = [np.random.uniform(-32.768, 32.768, DIM) for _ in range(pop_size)]  # je Individuum 20 DIM in gegebener range
    best_val = float('inf')

    for gen in range(MAX_ITER):
        scores = [ackley(ind) for ind in pop]
        current_best = float(np.min(scores))

        if current_best < best_val:
            best_val = current_best

        if (gen + 1) % PRINT_EVERY == 0:
            print(f"ga step {gen + 1}: f(x)= {best_val}")

        if best_val < TOL:
            print(f"ga step {gen + 1}: f(x)= {best_val}  <-- letzter Schritt")
            break

        idx = np.argsort(scores)[:pop_size // 2]  # sortieren der ersten (besten) 20 Individuen
        parents = [pop[i] for i in idx]

        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(parents, 2)
            mix_ratio = np.random.rand(DIM)  # for each DIM
            child = mix_ratio * p1 + (1 - mix_ratio) * p2
            child += np.random.normal(0, 0.5, DIM)  # zufälliges Rauschen
            child = np.clip(child, -32.768, 32.768)
            children.append(child)

        pop = children

    return best_val


de_results = []
ga_results = []

for i in range(RUNS):
    de_results.append(run_de(i))
    ga_results.append(run_ga(i))

print("\n** Ergebnisse **")
print("DE mean:", np.mean(de_results))
print("GA mean:", np.mean(ga_results))
print("DE std:", np.std(de_results))  # standard deviation also wie zuverlässig ist der mean
print("GA std:", np.std(ga_results))
