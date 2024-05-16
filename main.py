import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from typing import List

def q(x_values_array: List[float], alpha: float) -> float:
    n = len(x_values_array)
    # Unikamy użycia logarytmu dla wartości, które mogą prowadzić do problemów
    return sum((np.exp((i-1)/(n-1) * np.log(alpha)) if alpha > 0 else 0) *
               (np.exp(2 * np.log(x_values_array[i-1])) if x_values_array[i-1] > 0 else 0)
               for i in range(1, n+1))

def solver(q: callable, x0: List[float], alpha: float, learning_rate: float,
           max_num_iterations: int, epsilon: float = 1e-6, min_change_threshold: float = 1e-8) -> List[float]:
    x_current = np.array(x0, dtype=float)

    gradient_q = grad(q)

    all_q_values = []

    for i in range(max_num_iterations):
        q_value = q(x_current, alpha)
        all_q_values.append(q_value)

        grad_value = gradient_q(x_current, alpha)

        if np.any(np.isnan(grad_value)) or np.linalg.norm(grad_value) < epsilon or np.isnan(q_value):
            break

        if i > 0 and abs(all_q_values[-1] - all_q_values[-2]) < epsilon:
            break

        x_next = x_current - learning_rate * grad_value

        if np.any(np.isnan(x_next)) or np.any(np.isinf(x_next)):
            print(f"Overflow encountered in x_next at iteration {i+1}. Stopping...")
            break

        x_current = x_next

        # Sprawdź warunek na zbliżenie dwóch kolejnych wartości funkcji q
        if i > 0 and abs(all_q_values[-1] - all_q_values[-2]) < min_change_threshold:
            print(f"Local minimum found at iteration {i+1}. q(x) value: {all_q_values[-1]}")
            break

    return all_q_values

x_start_values = [100] * 10
alpha_values = [1, 10, 100]
learning_rates = [0.001, 0.01, 0.1]  # Test różnych wartości learning_rate
max_num_iterations = 10000

local_minima = {}

for alpha in alpha_values:
    local_minima[alpha] = {}
    for learning_rate in learning_rates:
        all_q_values = solver(q, x_start_values, alpha, learning_rate, max_num_iterations)
        local_minima[alpha][learning_rate] = all_q_values

# Wykresy dla różnych alpha i wszystkich learning_rate
for alpha in alpha_values:
    plt.figure(figsize=(12, 8))
    plt.xlabel('Iteration')
    plt.ylabel('q(x)')
    plt.title(f'Convergence of Gradient Descent for Alpha = {alpha}')

    for lr in learning_rates:
        plt.semilogy(local_minima[alpha][lr], label=f'Learning Rate = {lr}')

    plt.grid(True)
    plt.legend()
    plt.show()
