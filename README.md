Tests on a linear parabolic equation
$$u_{t}+\nabla u=\mu sin(2\pi*x)*sin(2\pi*y)$$
with boundary and initial conditions equal to 0 everywhere. Inspired from [this](https://github.com/RBniCS/RBniCS/tree/master/tutorials/06_thermal_block_unsteady).

|Method                                     |Train error|Test Error|Time   |
|-------------------------------------------|-----------|----------|-------|
|PINN                                       |3.5e-02    |3.5e-02   |5.4e+01|
|PODGalerkin(Nmax=20,netestedPOD=4)         |1.9e-05    |1.9e-05   |4.6e+01|
|PODGalerkin(Nmax=20)                       |1.9e-05    |1.9e-05   |4.6e+01|
|ReducedBasisSpectral(Nmax=20,PODGreedy=4)  |1.4e-07    |1.4e-07   |2.1e+01|
|ReducedBasisPOD(Nmax=20,PODGreedy=4)       |1.4e-07    |1.4e-07   |2.6e+01|
|Tree                                       |0.0e+00    |9.4e-03   |1.8e-02|
|GPR                                        |9.1e-07    |1.2e-06   |4.2e-02|
|RBF                                        |1.2e-15    |1.5e-07   |2.8e-02|
