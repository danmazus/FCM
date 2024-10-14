import sub_routine_1_Lv_mult
import sub_routine_2
import initialization

# Taking initializations from initialization file
D = initialization.D # Some Dense Matrix
v = initialization.v
M = initialization.M
n = initialization.n
LT = initialization.LT

# Lower Triangular and Vector Multiplication
M = sub_routine_1_Lv_mult.Lv_mult(LT, v, n)
print("Resulting Matrix M: ", M)
for r in M:
    print(r)