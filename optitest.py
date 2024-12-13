import os
import pandas as pd
import numpy as np

def read_transportation_problem(file_path):
    """
    Reads the transportation problem from a structured text file or CSV file.
    The file should include a cost matrix, supply row, and demand column or be
    structured with Supply Nodes, Demand Nodes, and Cost Matrix sections.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: supply_nodes, demand_nodes, cost_matrix
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, index_col=0)

        supply = data.iloc[:-1, -1].values  # Last column except last row
        demand = data.iloc[-1, :-1].values  # Last row except last column
        cost_matrix = data.iloc[:-1, :-1].values  # All except last row/column

        return list(supply), list(demand), cost_matrix

    else:  # For text file with structured format
        with open(file_path, 'r') as file:
            lines = file.readlines()

        supply_nodes = {}
        demand_nodes = {}
        cost_matrix = []

        section = None
        for line in lines:
            line = line.strip()
            if line == "Supply Nodes:":
                section = "supply"
            elif line == "Demand Nodes:":
                section = "demand"
            elif line == "Cost Matrix:":
                section = "cost"
            elif section == "supply" and line:
                node, supply = line.split(': ')
                supply_nodes[node] = int(supply)
            elif section == "demand" and line:
                node, demand = line.split(': ')
                demand_nodes[node] = int(demand)
            elif section == "cost" and line:
                costs = list(map(int, line.split()))
                cost_matrix.append(costs)

        supply = list(supply_nodes.values())
        demand = list(demand_nodes.values())

        return supply, demand, np.array(cost_matrix)

def northwest_corner_rule(supply, demand, cost):
    """Find initial feasible solution using Northwest Corner Rule."""
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))

    i, j = 0, 0
    while i < m and j < n:
        allocation[i, j] = min(supply[i], demand[j])
        if supply[i] < demand[j]:
            demand[j] -= supply[i]
            i += 1
        else:
            supply[i] -= demand[j]
            j += 1

    return allocation

def minimum_cost_method(supply, demand, cost):
    """Find initial feasible solution using Minimum Cost Method."""
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))

    cost_indices = [(i, j) for i in range(m) for j in range(n)]
    cost_indices.sort(key=lambda x: cost[x[0], x[1]])

    for i, j in cost_indices:
        if supply[i] == 0 or demand[j] == 0:
            continue
        allocation[i, j] = min(supply[i], demand[j])
        if supply[i] < demand[j]:
            demand[j] -= supply[i]
            supply[i] = 0
        else:
            supply[i] -= demand[j]
            demand[j] = 0

    return allocation

def minimum_row_cost_method(supply, demand, cost):
    """Find initial feasible solution using Minimum Row Cost Method."""
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))

    for i in range(m):
        while supply[i] > 0:
            min_cost = np.inf
            min_j = -1
            for j in range(n):
                if demand[j] > 0 and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    min_j = j
            alloc = min(supply[i], demand[min_j])
            allocation[i, min_j] = alloc
            supply[i] -= alloc
            demand[min_j] -= alloc

    return allocation

def vogels_approximation_method(supply, demand, cost):
    """Find initial feasible solution using Vogel's Approximation Method."""
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        penalties = []
        for i in range(m):
            if supply[i] > 0:
                row = sorted([cost[i, j] for j in range(n) if demand[j] > 0])
                penalties.append((row[1] - row[0] if len(row) > 1 else row[0], i, 'row'))
        for j in range(n):
            if demand[j] > 0:
                col = sorted([cost[i, j] for i in range(m) if supply[i] > 0])
                penalties.append((col[1] - col[0] if len(col) > 1 else col[0], j, 'col'))

        penalties.sort(reverse=True, key=lambda x: x[0])
        penalty, index, type_ = penalties[0]

        if type_ == 'row':
            i = index
            j = np.argmin([cost[i, j] if demand[j] > 0 else np.inf for j in range(n)])
        else:
            j = index
            i = np.argmin([cost[i, j] if supply[i] > 0 else np.inf for i in range(m)])

        allocation[i, j] = min(supply[i], demand[j])
        if supply[i] < demand[j]:
            demand[j] -= supply[i]
            supply[i] = 0
        else:
            supply[i] -= demand[j]
            demand[j] = 0

    return allocation

# Example Usage
file_path = 'example.csv'  # Update with your file path or text file path
supply, demand, cost_matrix = read_transportation_problem(file_path)

print("Cost Matrix:")
print(cost_matrix)
print("Supply:", supply)
print("Demand:", demand)

print("\nNorthwest Corner Rule Allocation:")
print(northwest_corner_rule(supply.copy(), demand.copy(), cost_matrix))

print("\nMinimum Cost Method Allocation:")
print(minimum_cost_method(supply.copy(), demand.copy(), cost_matrix))

print("\nMinimum Row Cost Method Allocation:")
print(minimum_row_cost_method(supply.copy(), demand.copy(), cost_matrix))

print("\nVogel's Approximation Method Allocation:")
print(vogels_approximation_method(supply.copy(), demand.copy(), cost_matrix))
