from dolfin import *
from mshr import *
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 30)

subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary = Boundary()
boundary.mark(boundaries, 1)

File("square.xml") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries
XDMFFile("square.xdmf").write(mesh)
XDMFFile("square_region.xdmf").write(subdomains)
XDMFFile("square_facet_region.xdmf").write(boundaries)