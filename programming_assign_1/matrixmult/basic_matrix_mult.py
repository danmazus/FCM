import random
#for i in range(len(x)):  # tells us how many rows are in x
#    for j in range(len(y[0])):  # tells us how many columns are in y
#        for k in range(len(y)):  # tells us how many rows are in y
#            result[i][j] += x[i][k] * y[k][
#                j]  # for x, i gets incremented first and for y, k is incremented first; x[i][k] is rows, y[k][j] is cols

#for r in result:
#    print(r)
n = int(input("Number of dimensions: "))
a = int(input("Lower Bound for Range of Random Numbers: "))
b = int(input("Upper Bound for Range of Random Numbers: "))

# Lower Triangular Matrix with 1 in the diagonal and below
LT = [[1 if i <= k else 0 for k in range(n)] for i in range(n)]
print("Lower Triangular Matrix L: ")
for r in LT:
    print(r)

# Upper Triangular Matrix with 1 in the diagonal and above
U = [[1 if i <= k else 0 for k in range(n)] for i in range(n)]
print("Upper Triangular Matrix U: ")
for r in U:
    print(r)


# Random Matrix Generator
RANMAT = [[random.randint(a, b) for k in range(n)] for i in range(n)]
print("Random Matrix RANMAT: ")
for r in RANMAT:
    print(r)

# Random Lower Triangular Matrix Generator
RANLMAT = [[random.randint(a, b) if i>= k else 0 for k in range(n)] for i in range(n)]
print("Random Lower Triangular Matrix RANLMAT: ")
for r in RANLMAT:
    print(r)

# Random Upper Triangular Matrix Generator
RANUMAT = [[random.randint(a, b) if i<= k else 0 for k in range(n)] for i in range(n)]
print("Random Upper Triangular Matrix RANUMAT: ")
for r in RANUMAT:
    print(r)

L = [[2] * n for i in range(n)] # Initializing a matrix of n dimensions specified above
print("Matrix L: ")
for r in L:
    print(r)

v = [[2] for i in range(n)]
print("Vector v: ")
for r in v:
    print(r)

M = [[0] * 1 for i in range(n)]
for i in range(len(L)):
    #for j in range(len(v[0])):
        for k in range(len(v)):
            M[i] += L[i][k] * v[k]

print("Matrix M: ")
for r in M:
    print(r)

