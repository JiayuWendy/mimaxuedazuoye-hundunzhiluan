import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce
import time
from memory_profiler import profile


def logistic_map(seed, mu, M, N):
    sequence = []
    x = seed
    for _ in range(M):
        x = mu * x * (1 - x)

    for _ in range(N):
        x = mu * x * (1 - x)
        sequence.append(x)

    return sequence


def generate_permutation(sequence):
    sorted_indices = np.argsort(sequence)
    permutation = np.argsort(sorted_indices)
    return permutation


def find_cycles(permutation):
    n = len(permutation)
    visited = [False] * n
    cycles = []

    for i in range(n):
        if not visited[i]:
            current = i
            cycle = []
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = permutation[current]
            cycles.append(cycle)

    return cycles


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def calculate_cycle_properties(cycles):
    cycle_lengths = [len(cycle) for cycle in cycles]
    unique_lengths = set(cycle_lengths)
    cycle_count = {length: cycle_lengths.count(length) for length in unique_lengths}
    total_cycle_length = reduce(lcm, cycle_lengths)

    return unique_lengths, cycle_count, total_cycle_length


@profile
def average_cycle_length(N, seeds, mu, M):
    total_length = 0
    for seed in seeds:
        sequence = logistic_map(seed, mu, M, N)
        permutation = generate_permutation(sequence)
        cycles = find_cycles(permutation)
        unique_lengths, cycle_count, total_cycle_length = calculate_cycle_properties(cycles)
        print("Seed:", seed)
        print("Unique Cycle Lengths:", unique_lengths)
        print("Number of Cycles for Each Length:", cycle_count)
        print("Total Cycle Length:", total_cycle_length)
        total_length += total_cycle_length

    return total_length / len(seeds)


@profile
def plot_average_cycle_length(N_values, seeds, mu, M):
    average_lengths = []
    for N in N_values:
        start_time = time.time()
        avg_length = average_cycle_length(N, seeds, mu, M)
        end_time = time.time()
        average_lengths.append(avg_length)
        print(f"N={N}, Average Cycle Length={avg_length}, Time taken={end_time - start_time:.4f} seconds")

    plt.plot(N_values, average_lengths)
    plt.xlabel('N')
    plt.ylabel('Average Cycle Length')
    plt.title('Average Cycle Length vs N')
    plt.grid(True)
    plt.show()


@profile
def test_seed_sensitivity(seed, mu, M, N, epsilon=1e-5):
    sequence1 = logistic_map(seed, mu, M, N)
    permutation1 = generate_permutation(sequence1)

    sequence2 = logistic_map(seed + epsilon, mu, M, N)
    permutation2 = generate_permutation(sequence2)

    different_positions = np.sum(permutation1 != permutation2)
    print(f"Sensitivity Test:\nSeed: {seed} and {seed + epsilon}")
    print(f"Different Positions: {different_positions} out of {N}")


@profile
def test_attack_resistance(seed, mu, M, N):
    sequence = logistic_map(seed, mu, M, N)
    permutation = generate_permutation(sequence)

    known_permutation = permutation[:N // 2]
    predicted_permutation = np.argsort(sequence)[:N // 2]

    correct_predictions = np.sum(known_permutation == predicted_permutation)
    print(f"Attack Resistance Test:\nSeed: {seed}")
    print(f"Correct Predictions: {correct_predictions} out of {N // 2}")


# Parameters
mu = 3.9
M = 1000
N = 100
seeds = np.random.rand(10)
N_values = range(10, 200, 2)

# 绘制平均循环长度与 N 的曲线
plot_average_cycle_length(N_values, seeds, mu, M)

# 种子敏感度测试
test_seed_sensitivity(seed=0.5, mu=3.9, M=1000, N=100)

# 对抗攻击能力测试
test_attack_resistance(seed=0.5, mu=3.9, M=1000, N=100)
