import bpy
import taichi as ti
import taichi.math as mti
import numpy as np
from math import acos

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

        self.compliance = 1e7 / self._dt**2
        self._bcompliance = 1e7 / self._dt**2
        self.relax_coeff_stretch = 1.
        self.relax_coeff_bending = 1.
        self.friction_coeff = 0.9

        self.is_bending = False
        self.ground = True

    def print_parameter(self):
        super().print_parameter()
        print(f" {'stretch compliance':<10}: {self.compliance}")
        print(f" {'bending compliance':<10}: {self._bcompliance}")
        print(f" {'relaxation stretch coeff':<10}: {self.relax_coeff_stretch}")
        print(f" {'relaxation bending coeff':<10}: {self.relax_coeff_bending}")
        print(f" {'frictioin coeff':<10}: {self.friction_coeff}")
        print(f" {'Bending':<10}: {self.is_bending}")
        print(f" {'ground':<10}: {self.ground}")
        print("------------------------------------------------------")
        return

    @property
    def compliance(self):
        return self._compliance
    @compliance.setter
    def compliance(self, new_compliance):
        self._compliance = new_compliance
        self._alpha_compliance = self._compliance / self._dt**2

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

        # Create list of involved point in the bending_constraint
        bending_cons = list()
        for e in range(len(self.edges)):
            bending_cons.append(list())
        for i, p in enumerate(obj.data.polygons):
            # trouve les voisins
            n_vert = len(p.vertices)
            # vertices id of the face
            verts = p.vertices
            # verts = [self.points[v_id] for v_id in p.vertices]
            # The edges of the face as a set of 2 vertice indices
            edges = [set([verts[i],verts[(i+1)%n_vert]]) for i in range(n_vert)]
            # List of edges id of the face
            edges_id = len(edges) * [None]
            for e_id, e in enumerate(self.edges):
                for _e_id, _e in enumerate(edges):
                    if set(e)==_e:
                        edges_id[_e_id] = e_id
            # edges_id = [e_id for e_id, e in enumerate(self.edges) if set(e) in edges]
            # edges_id is not sorted along edges
            for e_id, e in zip(edges_id, edges):
                cons_point=bending_cons[e_id]
                e_list = list(e)
                cons_point.append(e_list[0]) if e_list[0] not in cons_point else None
                cons_point.append(e_list[1]) if e_list[1] not in cons_point else None
                f_list = list(set(verts)-e)
                cons_point.append(f_list[0])
                # bending_cons[e_id] = cons_point
        self.bending_point = [ids for ids in bending_cons if len(ids)>=4]
        self.n_bending_cons = len(self.bending_point)
        del[bending_cons]

        # Compute rest Bending angle
        # TODO
        b0 = list()
        for b in self.bending_point:
            id_1, id_2, id_3, id_4 = b
            real_p1 = self.points[id_1]
            p2 = self.points[id_2] - real_p1
            p3 = self.points[id_3] - real_p1
            p4 = self.points[id_4] - real_p1
            n1 = np.cross(p2, p3)
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(p2, p4)
            n2 = n2 / np.linalg.norm(n2)
            d = acos(np.dot(n1, n2))
            b0.append(d)
        self.bending_cons_b0 = np.array(b0, dtype=np.single)

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

        self.stretch_cons  = ti.Vector.field(2, dtype=int, shape=self.n_edge)
        self.bending_cons  = ti.Vector.field(4, dtype=int, shape=self.n_bending_cons)
        self.l0    = ti.field(float, shape=self.n_edge)
        self.b0    = ti.field(float, shape=self.n_bending_cons)
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
        self.stretch_cons.from_numpy(np.array(self.edges, dtype=np.int32))
        self.l0.from_numpy(self.edges_l0)

        self.bending_cons.from_numpy(np.array(self.bending_point, dtype=np.int32))
        self.b0.from_numpy(self.bending_cons_b0)


    def step_forward(self):
        self.predict()
        if self.is_bending:
            self.bending_compute()
        self.stretch_compute()
        self.update_field()

    @ti.kernel
    def predict(self):
        for i in self.x:
            v = self.v[i] + self._dt * self.gravity
            self.x_tmp[i] = self.x[i] + v * self._dt
            self.dx[i] = 0.

    @ti.kernel
    def stretch_compute(self):
        for e in self.stretch_cons:
            id_1, id_2 = self.stretch_cons[e][0], self.stretch_cons[e][1] 
            v0, v1 = self.x_tmp[id_1], self.x_tmp[id_2]
            vec = v1-v0
            dist = vec.norm()
            constraint = dist - self.l0[e]
            lmbda = constraint / (self.mass + self._compliance) / self.relax_coeff_stretch
            self.dx[id_1] += lmbda / 2 * vec.normalized()
            self.dx[id_2] -= lmbda / 2 * vec.normalized()

    @ti.kernel
    def bending_compute(self):
        for e in self.bending_cons:
            # Get x values
            id_1, id_2, id_3, id_4 = self.bending_cons[e]
            real_p1 = self.x[id_1]
            p2 = self.x[id_2] - real_p1
            # p3 = (self.x[id_3]+self.x[id_4])/2 - real_p1
            # p4 = (self.x[id_5]+self.x[id_6])/2 - real_p1
            p3 = self.x[id_3] - real_p1
            p4 = self.x[id_4] - real_p1

            # Compute constraint value
            n1 = mti.normalize(mti.cross(p2, p3))
            n2 = mti.normalize(mti.cross(p2, p4))
            d = mti.dot(n1, n2)
            d = ti.math.clamp(d, -1., 1.)

            cons_val = mti.acos(d) - self.b0[e]

            # Compute Grad C for each points
            q3 = (mti.cross(p2, n2) + mti.cross(n1, p2) * d ) / n1.norm()
            q4 = (mti.cross(p2, n1) + mti.cross(n2, p2) * d ) / n2.norm()
            q2 = - (mti.cross(p3, n2) + mti.cross(n1, p3) * d ) / n1.norm() - (mti.cross(p4, n1) + mti.cross(n2, p4) * d ) / n2.norm()
            q1 = - q2 - q3 - q4
            q_sum = self.mass * (mti.dot(q1, q1) + mti.dot(q2, q2) + mti.dot(q3, q3) + mti.dot(q4, q4))
            # q_sum = self.mass * (q1.norm_sqr()+q2.norm_sqr()+q3.norm_sqr()+q4.norm_sqr())
            
            # lmbda
            lmbda = - (mti.sqrt(1-d*d) * cons_val) / (q_sum + self._bcompliance) / self.relax_coeff_bending

            # print("")
            # print(n1, n2)
            # print("cos=", d)
            # print("cons=", cons_val)
            # print("lambda=", lmbda)
            # print(f"dx({id_1})", self.mass * lmbda * q1)
            # print(f"dx({id_2})", self.mass * lmbda * q2)
            # print(f"dx({id_3})", self.mass * lmbda * q3)
            # print(f"dx({id_4})", self.mass * lmbda * q4)

            # Update correction displacement
            self.dx[id_1] += self.mass * lmbda * q1
            self.dx[id_2] += self.mass * lmbda * q2 
            self.dx[id_3] += self.mass * lmbda * q3
            self.dx[id_4] += self.mass * lmbda * q4

    @ti.kernel
    def update_field(self):
        for i in self.x:
            if i!=0 and i!=1:
                x_m1 = self.x[i]
                new_x = self.x_tmp[i] + self.dx[i] # relax_coeff is the mean constraint by point
                v = (new_x - x_m1) / self._dt
                if self.ground:
                    if new_x.z<0:
                        new_x.z = -new_x.z
                        v.x = ti.sqrt(self.friction_coeff) * v.x
                        v.y = ti.sqrt(self.friction_coeff) * v.y
                        v.z = -v.z
                self.x[i] = new_x# self.x_tmp[i] + self.dx[i]/4. * self.relax_coeff
                self.v[i] = v # (self.x[i] - x_m1) / self._dt
