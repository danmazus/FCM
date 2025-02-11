import numpy as np

"""
This task will perform the subtasks for the f_3 function:
    1.  The interpolating problem that the given polynomial solves on the uniform mesh points and Chebyshev
        points of the first and second kind, i.e., y_i = f(x_i), 0 <= i <= m where m is n for f_3(x). Also,
        include some tests that use more points than necessary to reproduce the polynomial f_3(x). For f_3(x)
        choose at least two different degrees that are greater than 20 for each mesh type. For f_3(x) include
        n = 29 (30 points) to compare to the results in Higham's paper.
    2.  For each of the degrees used for the uniform and Chebyshev meshes, determine the conditioning by
        evaluating k(x,n,y( and k(x,n,1) for a <= x <= b and summarize them appropriately using \Lambda_n
        and H_n along with appropriate statistics.
    3.  Assess the accuracy and stability of the single precision codes using the appropriate bounds from the
        notes and literature on the class webpage and the values generated in determining the conditioning in
        the subtask above. (The condition numbers and "exact" values of the interpolating polynomial for accuracy
        and stability assessment should be done in double precision.) This should be done for
            - Barycentric form 2 of the polynomial
            - Newton form with the mesh points in increasing order, decreasing order, and satisfying the Leja
            ordering conditions
        You should provide plots similar to those in Higham's paper for a small number of illustrative examples
        with 30 points for the uniform mesh and the Chebyshev points of the first kind. The other results should be
        summarized to comment on the accuracy and stability. Note that Higham's experiments are run in double precision
        and compared to "exact" values from Matlab's 50 digit symbolic arithmetic toolbox, so you will not see exactly
        the same behavior.
"""

