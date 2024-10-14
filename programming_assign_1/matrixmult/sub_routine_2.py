from random import randint

# Function Definition for Unit Lower Triangular and Vector Multiplication stored in Compressed Columns
def Lv_mult_cr(crv, v, n):
    # Initialization for Resulting Vector
    m = [0] * n
    # Index to track position in CRV
    index = 0

    # Loops over rows of M
    for i in range(n):
        for k in range(i):
            # This is all non-diagonal elements below the diagonal in the given row i
            m[i] = crv[index] * v[k]
            index += 1
        # Adding the diagonal element -- this gave me trouble to figure out
        m[i] += v[i]

    return m

n = int(input("Number of dimensions: "))
a = int(input("Lower Bound for Random Number in Vector: "))
b = int(input("Upper Bound for Random Number in Vector: "))

# Initializing Unit Lower Triangular Matrix
ULT = [[0] * n for i in range(n)]

# For loop to create random elements in the Unit Lower Triangular Matrix ULT
for i in range(n):
    for k in range(n):
        if i == k:
            ULT[i][k] = 1
        elif i < k:
            ULT[i][k] = 0
        else:
            ULT[i][k] = randint(a, b)

# Having to calculate S given n dimensions
crv_size = ((n * (n - 1)) // 2)

# Initializing CRV with the correct size given the above calculation
crv = [0] * crv_size

# Initializing the index for the CRV vector
index = 0

# Looping over the rows and columns of ULT to replace the 0s in crv with the below non-diagonal elements of row i
for i in range(n):
    for k in range(i):
        crv[index] = ULT[i][k]
        index += 1


print("The Matrix ULT is: ")
for r in ULT:
    print(r)

print("Resulting CRV is: ", crv)

# Initializing for Vector v
v = [0] * n
for i in range(n):
    v[i] = randint(a, b)
print("The Vector v is: ", v)

w = Lv_mult_cc(crv, v, n)

print("Resulting Vector w is: ", w)
