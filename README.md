# FCM
Programming Assignments and Custom Functions used in FCM Class
This repository is not open for pull requests. This is to document projects I have done within my FCM Class. If you want to make a contribution or have a better idea of how to write the code, please submit a review or pull request review so I can better the code itself.

## My Package
This includes a function storage file that can be called and installed locally in the desired location. This is storing functions used within the programming assignments and will be updated as new functions are created.

## Programming Assignment 1
Programming Assignment 1 contains 4 subroutines that are designed to do Matrix-Vector multiplication with different storage of the matrices and Matrix-Matrix multipication within a single 2D array. These were mainly used for unit lower triangular matrices and upper triangular matrices that were not conditioned in this assignment.

## Programming Assignment 2
Programming Assignment 2, in Program 2 folder, contains 3 subroutines and a tester with empirical tests. The first routine does LU Factorization with no pivoting and partial pivoting depending on the given routine specified in the user when ran. The other 2 routines are for solving Ax = b where A = LU and now accomplishing triangular solves, Ly = b then Ux = y to solve for x. These can be changed around for what can be accomplished. The tester is found in the LU Factorization subroutine after the functions were defined. These have yet to be added to my_package which contains a function storage of different custom functions used in the assignments.

## Least Squares
This folder has the Householder Reflector subroutine which computes the Householder vector which transform an input vector and outputs the transformed vector and the scalar value that led to the transformation. This also has the tester that was used in class to test this routine using Least Squares. This was translated from in class code from Dr. Gallivan from Matlab to Python for both the subroutine and the Householder Reflector Routine.