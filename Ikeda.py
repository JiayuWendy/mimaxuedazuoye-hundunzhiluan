import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce


def ikeda_map(seed, M, N):
    a = 0.918
    b = 0.025
    sequence = []
    x, y = seed
    for _ in range(M):
        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x_new = 1 + a * (x * np.cos(t) - y * np.sin(t))
        y = b + a * (x * np.sin(t) + y * np.cos(t))
        x = x_new

    for _ in range(N):
        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x_new = 1 + a * (x * np.cos(t) - y * np.sin(t))
        y = b + a * (x * np.sin(t) + y * np.cos(t))
        x = x_new
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


def average_cycle_length(N, seeds, M):
    total_length = 0
    for seed in seeds:
        sequence = ikeda_map(seed, M, N)
        permutation = generate_permutation(sequence)
        cycles = find_cycles(permutation)
        unique_lengths, cycle_count, total_cycle_length = calculate_cycle_properties(cycles)
        print("Seed:", seed)
        print("Unique Cycle Lengths:", unique_lengths)
        print("Number of Cycles for Each Length:", cycle_count)
        print("Total Cycle Length:", total_cycle_length)
        total_length += total_cycle_length

    return total_length / len(seeds)


def plot_average_cycle_length(N_values, seeds, M):
    average_lengths = []
    for N in N_values:
        avg_length = average_cycle_length(N, seeds, M)
        average_lengths.append(avg_length)

    plt.plot(N_values, average_lengths)
    plt.xlabel('N')
    plt.ylabel('Average Cycle Length')
    plt.title('Average Cycle Length vs N')
    plt.grid(True)
    plt.show()


# Parameters
M = 1000
N = 100
seeds = np.random.rand(10, 2)  # 生成二维种子
N_values = range(10, 200, 2)

# 绘制平均循环长度与 N 的曲线
plot_average_cycle_length(N_values, seeds, M)
