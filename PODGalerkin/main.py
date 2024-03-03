# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *    
from time import time
import numpy as np

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
reduction_method.set_Nmax(20)
reduction_method.set_tolerance(1e-6)

# 5. Perform the offline phase
lifting_mu = (0.5,)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(100)
reduced_problem = reduction_method.offline()

training_param=[(0.7289120207996115,), (0.25409463628131845,), (0.11009024695349051,), (0.9557287080549567,), (0.7111841098771446,), (0.5639913111779458,), (0.317057353131431,), (0.4173116530978206,), (0.951601632294682,), (0.8187215764570569,), (0.6265946679318678,), (0.9157032519815728,), (0.11797774345327781,), (0.5139512809568915,), (0.8928363594937064,), (0.6173035001673071,), (0.4701805002649244,), (0.9853589932547693,), (0.3458669798645811,), (0.1781860793536314,), (0.44820690775673344,), (0.2224407887449683,), (0.5257490624696358,), (0.9834568326702106,), (0.20612809566657997,), (0.21763296080042366,), (0.681764970719294,), (0.7903778599048281,), (0.20091542682148972,), (0.5863304351949501,), (0.9377992293962237,), (0.5422700980792073,), (0.14607551415736647,), (0.3386767115889571,), (0.6107070535689356,), (0.1817910418370403,), (0.6840203319811349,), (0.5828853290127993,), (0.1169948996694129,), (0.8000870971191111,), (0.4346126279215371,), (0.49256657755736755,), (0.581869052012907,), (0.7501792068145007,), (0.7758667624626573,), (0.8888947165257599,), (0.4988312766650844,), (0.6791047884868425,), (0.8892165976415137,), (0.35821250190175413,), (0.9171422588046395,), (0.5536799648633998,), (0.6915333466712602,), (0.15473974073568258,), (0.3446104126774485,), (0.9632901291695781,), (0.3184032666046604,), (0.14390268594626723,), (0.47635129415356003,), (0.8729330540271749,), (0.24343061755484577,), (0.5639601307039651,), (0.7905645792763382,), (0.6159551493515403,), (0.7488487610971467,), (0.4983361170766347,), (0.280592855452091,), (0.5457239559390114,), (0.12504423371464754,), (0.44408027484140156,), (0.3595512110416761,), (0.3254797474200507,), (0.3011557475707408,), (0.9901286151456867,), (0.3060223952946383,), (0.5222035030645408,), (0.9243808004735911,), (0.5509245973761588,), (0.7917189266517366,), (0.622551061659853,), (0.8648550932650503,), (0.8000694177193837,), (0.8816243454370987,), (0.14981761718185166,), (0.779490167269205,), (0.4016123425883861,), (0.6034140116439618,), (0.15846220967836758,), (0.9586851002312624,), (0.44371555334021406,), (0.4950826771244593,), (0.7361237473129891,), (0.9493696989038557,), (0.7819922641912838,), (0.9302579684995561,), (0.6626264510457329,), (0.7618815099532775,), (0.3957944159321374,), (0.5159701975474963,), (0.22450716754270023,)]
testing_param=[(0.29921321325146183,), (0.19589565374588647,), (0.6870708850241003,), (0.28750077782961203,), (0.7730346294166374,), (0.1781230530051563,), (0.4745575269808454,), (0.4883828890964378,), (0.6007696767615336,), (0.6112064988749245,), (0.25525925748182066,), (0.21551141596049678,), (0.9570013432925175,), (0.15133908704859325,), (0.8268651826681239,), (0.44949056782395613,), (0.1561199710524522,), (0.9997912649646767,), (0.31027479463591673,), (0.13888707422197313,), (0.10412082321184302,), (0.23429711069230544,), (0.2233860562908993,), (0.42848245542022545,), (0.6643829536391305,), (0.45502399322567166,), (0.6131281988427693,), (0.4440440487242798,), (0.10784657140825446,), (0.7922875823288323,), (0.2877821909333719,), (0.36738366031078795,), (0.9946463797802948,), (0.10925270830571944,), (0.6778120584069944,), (0.5656871714119396,), (0.482894106271778,), (0.5801493172250752,), (0.9616384378825654,), (0.7042618992932801,), (0.11748887452615706,), (0.8559653919946534,), (0.602980110233814,), (0.9725749463015676,), (0.16620197715567842,), (0.39297910479932485,), (0.3430973435600633,), (0.9755089043886332,), (0.8511956290207364,), (0.3107244462018923,), (0.9593258172467718,), (0.14806847028297013,), (0.6751453590100697,), (0.17893448050398003,), (0.9088987416116028,), (0.41326367174257195,), (0.5311653492666834,), (0.2529319588938637,), (0.5818303701086817,), (0.9180073440319484,), (0.1861232384276716,), (0.506637032815434,), (0.6308665857154211,), (0.10921292693428543,), (0.43762863233648364,), (0.5255682423639023,), (0.9154989219324361,), (0.8503896223787778,), (0.6291322039276639,), (0.8634202394951486,), (0.6194965013078241,), (0.6111407408280146,), (0.914957795778894,), (0.2568804049343961,), (0.9873342223263551,), (0.1020682876586104,), (0.6588780378764446,), (0.26687509266302,), (0.4131268050109126,), (0.6425875305679295,), (0.31028164776877903,), (0.28541561054173614,), (0.35603185633081735,), (0.4695276979031435,), (0.5712361795644239,), (0.5315034870177066,), (0.7459666033051982,), (0.4576823814180935,), (0.3733997303772826,), (0.7397165479675918,), (0.6698724571556885,), (0.36834207534429886,), (0.687249342191697,), (0.5687442092622955,), (0.12503423915750195,), (0.9800117488829487,), (0.6374058971873267,), (0.9520730304358185,), (0.12165128213142148,), (0.40401316992931724,)]

start=time()
for i in range(100):
    reduced_problem.set_mu(training_param[i])
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_train_{}".format(i))

for i in range(100):
    reduced_problem.set_mu(testing_param[i])
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_test_{}".format(i))

end=time()

np.save("time.npy",end-start)