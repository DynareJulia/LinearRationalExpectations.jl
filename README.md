WORK IN PROGRESS

``LinearRationalExpectations`` provides tools to solve economic linear
rational expectations models.

Linear rational expectations models have the general form

$$E_t \{ A y_{t+1} + B y_t + C y_{t-1} + D u_t + e\} = 0$$

The deterministic steady state of the model is defined as

$$\bar y = -(I - A - B - C)^{-1}e$$

The solution takes the form

$$y_t - \bar y= G_y (y_{t-1} - \bar y) + G_u u_t$$

where $G_y$  is the solution of the polynomial matrix equation

$$A G_y G_y + B G_y + C = 0$$

Two different algorithms are provided by package
``PolynomialMatrixEquations``:
one based on generalized Schur decomposition and one based on cyclic
reduction.

$$G_u = -(A G_y + B)^{-1}Du_t$$

In addition the ``LinearRationalExpectations`` package provides
functions to reduce the problem size by eliminating static variables.
