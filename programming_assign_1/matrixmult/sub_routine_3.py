from random import randint

# Function definition for a 2-bandwidth banded matrix and vector product
def Lv_mult_banded(vb, v, n):
    # Initialization for Resulting Vector
    m = [0] * n

    # Loops over rows of m
    for i in range(n):

        # Multiplying the diagonal of 1 * v for each i
        m[i] += v[i]

        # Checking to see if 1st sub-diagonal exists
        if i >= 1:
            m[i] += vb[1][i - 1] * v[i - 1]

        # Checking to see if 2nd sub-diagonal exists
        if i >= 2:
            m[i] += vb[0][i - 1] * v[i - 2]

    return m

n = int(input("Number of dimensions: "))
a = int(input("Lower Bound for Random Number in Vector: "))
b = int(input("Upper Bound for Random Number in Vector: "))

# Initializing Unit Lower Triangular Matrix
#ULT = [[0] * n for i in range(n)]

# For loop to create random elements in the Unit Lower Triangular Matrix ULT
#for i in range(n):
 #   for k in range(n):
  #      if i == k:
   #         ULT[i][k] = 1
    #    elif i < k:
     #       ULT[i][k] = 0
      #  else:
       #     ULT[i][k] = randint(a, b)

#print("The Matrix ULT is: ")
#for r in ULT:
#    print(r)

# Initializing Vector v with random numbers
v = [0] * n
for i in range(n):
    v[i] = randint(a, b)
print("The Vector v is: ", v)

# Initialization of 2D array named brv for Banded storage
brv = [[0] * n for i in range(n)]

# Looping over the rows and columns of ULT to replace the 0s in brv with the 2 sub-diagonals
# This is now creating a matrix brv that has zeros everywhere except for the 2 sub-diagonals

# This is now creating a matrix brv that has zeros everywhere except for the 2 sub-diagonals
for i in range(n):
    for k in range(i + 1):
        if i - 1 == k:
            brv[i][k] = randint(a, b)            # ULT[i][k]
        elif i - 2 == k:
            brv[i][k] = randint(a, b)            # ULT[i][k]
        elif i == k:
            brv[i][k] = 1                        # ULT[i][k]
        else:
            brv[i][k] = 0

print("The Matrix brv is: ")
for r in brv:
    print(r)

# We now want to put the sub-diagonals into their own arrays
# we must create a new 2D array that will store this
vb = [[0] * (n - 1) for i in range(2)]

for i in range(n): # rows
    for k in range(i): #columns
        if i - 2 == k:
            vb[0][k] = brv[i][k]
        elif i - 1 == k:
            vb[1][k] = brv[i][k]
        else:
            k += 1

# Shifting the Zero to the front of the array to show that there is a missing value because of the bands
for i in range(1):
    vb[i] = [0] + vb[i][:-1]

print("The Matrix vb is: ")
for r in vb:
    print(r)

w = Lv_mult_banded(vb, v, n)

print("The Vector w is: ", w)