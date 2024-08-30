import bpy
import taichi as ti
import numpy as np

from solver import Solver


@ti.data_oriented
class PositionBasedDynamic(Solver):
    """
    Description: 
    -------------
    """
    def __init__(self, arch) -> None:
        super().__init__(arch)
        self.name = "XPBD"

<<<<<<< HEAD
        self.compliance = 2e-7 / self._dt**2
=======
        self.compliance = 0.0 / self._dt**2
        self.relax_coeff = 1.

        self.bending_springs = False
>>>>>>> 253f263 (XPBD)

    def print_parameter(self):
        super().print_parameter()
        print(f" {'compliance':<10}: {self.compliance}")
<<<<<<< HEAD
=======
        print(f" {'relaxation coeff':<10}: {self.relax_coeff}")
        print(f" {'Bending':<10}: {self.bending_springs}")
>>>>>>> 253f263 (XPBD)
        print("------------------------------------------------------")
        return

    @property
    def compliance(self):
        return self._compliance
    @compliance.setter
    def compliance(self, new_compliance):
        self._compliance = new_compliance
        self._alpha_compliance = self._compliance / self._dt**2

<<<<<<< HEAD
    @property
    def mass(self):
        return self._mass
    @mass.setter
    def mass(self, new_mass):
        self._mass = new_mass
        self._invmass = 1./self._mass
        print(self._invmass)

=======
>>>>>>> 253f263 (XPBD)
    def initialize_from_obj(self, obj: bpy.types.Object):
        # Get Object sizes
        self.n      = len(obj.data.vertices)
        self.n_edge = len(obj.data.edges) # sum([len(e) for e in obj.data.edges], 0)
        self.n_prim = len(obj.data.polygons) # (self.n - 1) * (self.n - 1) * 2

        # Read points
        points = list()
        for v in obj.data.vertices:
            points.append([v.co.x, v.co.y, v.co.z])
        self.points = np.array(points, dtype=np.single)
        del[points]

        # Read edges
        edges = list()
        for e in obj.data.edges:
            edges.append([e.vertices[0], e.vertices[1]])
        self.edges = edges # np.array(edges, dtype=np.int32)
        del[edges]

        # Compute edges rest length
        edges_l0 = list()
        for e in self.edges:
            v1, v2 = self.points[e[0]], self.points[e[1]]
            l0 = np.linalg.norm(v2-v1)
            edges_l0.append(l0)
        self.edges_l0 = np.array(edges_l0, dtype=np.single)
        del[edges_l0]

        # Send to GPU fields
        ti.init(self._arch)
        self.create_fields()

        self.fill_points()
        self.fill_velocity()
        self.fill_constraints()

        self.isnot_init = False
        self.print_mesh_parameter()
        pass

    def create_fields(self):
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.x_tmp = ti.Vector.field(3, dtype=float, shape=self.n)
        self.v     = ti.Vector.field(3, dtype=float, shape=self.n)

        self.dx = ti.Vector.field(3, dtype=float, shape=self.n)

        self.cons  = ti.Vector.field(2, dtype=int, shape=self.n_edge)
        self.l0    = ti.field(float, shape=self.n_edge)
        pass

    def reset(self):
        self.fill_points()
        self.fill_velocity()
        pass

    def fill_points(self):
        self.x.from_numpy(self.points)
    
    @ti.kernel
    def fill_velocity(self):
        for i in self.v:
            self.v[i] = [0, 0, 0]

    def fill_constraints(self):
        self.cons.from_numpy(np.array(self.edges, dtype=np.int32))
        self.l0.from_numpy(self.edges_l0)


    def step_forward(self):
        self.predict()
        self.correction()
        self.update_field()

    @ti.kernel
    def predict(self):
        for i in self.x:
            v = self.v[i] + self._dt * self.gravity
            self.x_tmp[i] = self.x[i] + v * self._dt
            self.dx[i] = 0.

    @ti.kernel
    def correction(self):
        for e in self.cons:
            id_1, id_2 = self.cons[e][0], self.cons[e][1] 
            v0, v1 = self.x_tmp[id_1], self.x_tmp[id_2]
            vec = v1-v0
            dist = vec.norm()
            constraint = dist - self.l0[e]
<<<<<<< HEAD
            lmbda = constraint / (self._invmass + self._compliance)
=======
            lmbda = constraint / (self.mass + self._compliance)
>>>>>>> 253f263 (XPBD)
            self.dx[id_1] += lmbda / 2 * vec.normalized()
            self.dx[id_2] -= lmbda / 2 * vec.normalized()

    @ti.kernel
    def update_field(self):
        for i in self.x:
            if i!=0 and i!=1:
                x_m1 = self.x[i]
<<<<<<< HEAD
                self.x[i] = self.x_tmp[i] + self.dx[i]/8.
=======
                self.x[i] = self.x_tmp[i] + self.dx[i]/4. * self.relax_coeff
>>>>>>> 253f263 (XPBD)
                self.v[i] = (self.x[i] - x_m1) / self._dt