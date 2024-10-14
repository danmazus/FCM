n = int(input("Number of dimensions: "))
#a = int(input("Lower Bound for Range of Random Numbers: "))
#b = int(input("Upper Bound for Range of Random Numbers: "))

# Initializing L as a dense matrix
D = [[2] * n for i in range(n)]  # Initializing a matrix of n dimensions specified above
print("Matrix D: ")
for r in D:
    print(r)

# Lower Triangular Matrix Initialization as LT
LT = [[1 if i >= k else 0 for k in range(n)] for i in range(n)]
print("Lower Triangular Matrix L: ")
for r in LT:
    print(r)

# Initializing v as a vector
v = [[2] * n]
print("Vector v: ")
for r in v:
    print(r)

# Initializing M as the resulting matrix
M = [[0] * 1 for i in range(n)]

