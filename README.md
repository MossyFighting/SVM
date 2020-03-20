# Support Vector Machine (SVM)
## Detail of Support Vector Machine can be found at following link: [How to implement the SVM from scratch](https://github.com/MossyFighting/SVM/blob/master/report.pdf)

## Some notes:
In this project, A **support vector machine from Scratch** is implemented:
  1. Using dua-form of support vector machine optimization.
  2. SVM optimization is cast as a **convex optimization**.
  3. The **cvxpy** is used to optimize and obtain the lagrange multipliers, then support vectors are found.
  4. Some kernels are used in different examples such as: **linear kernel**, **polynomial kernel**, and **radial basis kernel**.
  
## Some results 
### Polynomial kernel:
In the example using polynomial kernel, supposing there are two classes that are linearly non-separable. Then, there is no any straight line can separate the two given classess.

![Two classes need to seperate](https://github.com/MossyFighting/SVM/blob/master/images/poly_no_boundary.png)

The polynomial has been used and the result show the green dash boundary that succeed in separating two classes. See the below figure:

![Boundary to seperate](https://github.com/MossyFighting/SVM/blob/master/images/poly.png)

In order to obtained the green dash boundary, the polynomial order degree = 5, and C = 0.2 are used.
### Radial basis kernel:
In the example below, the data is generating using **XOR** function, using numpy library **logic_xor**. The result shows the boundary with green dash line to sucessufully separate two given classes.

![Radial basis kernel to seperate](https://github.com/MossyFighting/SVM/blob/master/images/radial_basis.png)

In the example, the gamma = 0.5, and soft-margin constant value is used with value of C = 1.0


## Some notice:
### Advangtages
- It is good to have a small tool to visualize how to classify classes with many variable to tune, such as: soft-margin constant, gamma constant in radial basis function kernel, and so on.
- Some results show the SVM is able to solve the classification.
- The SVM classification can cast as convex optimization with inequality constraints and the results show the optimum separator.

### Disavantages:
- In this small software, the convex optimization is using cvxpy, and quadratic programming solver is used. However, this solver using matrix computation, then if the data is large, it is very slow, and it is efficient in reality when the data are very large.
- So, the disadvantage can be resolved when using another convex optimization, such as, SMO, or so on. 
