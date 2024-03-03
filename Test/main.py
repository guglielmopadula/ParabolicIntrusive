# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


class UnsteadyThermalBlock(ParabolicCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        ParabolicCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Store the initial condition expression
        self.ic = Constant(0.)
        self.bc = Constant(0.)
        self.f = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", element=self.V.ufl_element())

    # Return custom problem name
    def name(self):
        return "ParabolicLinear"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        t = self.t
        if term == "m":
            theta_m0 = 1.
            return (theta_m0, )
        elif term == "a":
            theta_a0 = 0.1
            return (theta_a0,)
        elif term == "f":
            theta_f0 = mu[0]
            return (theta_f0,)
        elif term == "dirichlet_bc":
            theta_bc0 = - mu[0]
            return (theta_bc0,)
        elif term == "initial_condition":
            theta_ic0 = - mu[0]
            return (theta_ic0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "m":
            u = self.u
            m0 = u * v * dx
            return (m0, )
        elif term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v)) * dx
            return (a0,)
        elif term == "f":
            f = self.f
            f0 = f* v * dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, self.bc, self.boundaries, 1)]
            return (bc0,)
        elif term == "initial_condition":
            ic0 = project(self.ic, self.V)
            return (ic0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        elif term == "projection_inner_product":
            u = self.u
            x0 = u * v * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("../data/square.xml")
subdomains = MeshFunction("size_t", mesh, "../data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "../data/square_facet_region.xml")


# 2. Create Finite Element space (Lagrange P1, two components)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the UnsteadyThermalBlock class
problem = UnsteadyThermalBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 1)]
problem.set_mu_range(mu_range)
problem.set_time_step_size(0.05)
problem.set_final_time(1)

# 4. Prepare reduction with a POD-Galerkin method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20, nested_POD=4)
reduction_method.set_tolerance(1e-6, nested_POD=1e-3)

# 5. Perform the offline phase
lifting_mu = (0.5,)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(100)
reduced_problem = reduction_method.offline()

