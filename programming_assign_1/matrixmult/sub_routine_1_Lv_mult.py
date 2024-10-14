from random import randint
import matplotlib.pyplot as plt

# Function Definition for Unit Lower Triangular Matrix and vector
def Lv_mult(ULT, v, n):
    M = [[0] for i in range(n)]

    # Loops over rows of M
    for i in range(n):
        for k in range(n):
            M[i][0] += ULT[i][k] * v[k][0]

    return M

# Function Definition for Random Unit Lower Triangular Matrices

# Function Definition for Random vectors

n = int(input("Number of dimensions: "))
a = int(input("Lower Bound for Random Number in Vector: "))
b = int(input("Upper Bound for Random Number in Vector: "))
num_runs = int(input("Number of runs: "))

# Initializing Unit Lower Triangular Matrix
ULT = [[0] * n for i in range(n)]
for i in range(n):
    for k in range(n):
        if i == k:
            ULT[i][k] = 1
        elif i < k:
            ULT[i][k] = 0
        else:
            ULT[i][k] = randint(a, b)

print("Unit Lower Triangular Matrix ULT: ")
for r in ULT:
    print(r)

# Initializing for Vector v
v = [[randint(a, b)] for i in range(n)]
print("Vector v: ")
for r in v:
    print(r)

M = Lv_mult(ULT, v, n)

print("Vector M: ")
for r in M:
    print(r)


def run_experiments(num_runs, n):
    results = []  # To store results of w from each run
    for i in range(num_runs):
        L = ULT
        vec = v
        w = Lv_mult(L, vec, n)
        results.append(w)
    return results


# Step 5: Plot the results
def plot_results(results):
    for i, run in enumerate(results):
        # Flatten the 2D list `w` (turns [[2], [3], [1], [4]] into [2, 3, 1, 4])
        flat_w = [row[0] for row in run]
        plt.plot(flat_w, label=f'Run {i + 1}')

    plt.title('Results of Multiple Runs (w = Lv)')
    plt.xlabel('Index of Vector w')
    plt.ylabel('Values of w')
    plt.legend()
    plt.show()

results = run_experiments(num_runs, n)
plot_results(results)