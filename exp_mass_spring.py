import bpy
import taichi as ti
import numpy as np

from solver import Solver

@ti.data_oriented
class ExplicitMassSpring(Solver):
    """
    Description: 
    -------------
    """

    def __init__(self, arch) -> None:
        super().__init__(arch)
        self.name = "Explicit Mass Spring"

        # Spring properties
        self.spring_rigidity = 1.7e5
        self.spring_damping = 1e9
        self.air_drag = 8 # 1.5

        self.friction_coeff = 0.70

        self.bending_springs = True

    def print_parameter(self):
        super().print_parameter()
        print(f" {'rigidity':<10}: {self.spring_rigidity}")
        print(f" {'damping':<10}: {self.spring_damping}")
        print(f" {'drag':<10}: {self.air_drag}")
        print(f" {'Bending':<10}: {self.bending_springs}")
        print("------------------------------------------------------")
        return

    def initialize_from_obj(self, obj: bpy.types.Object):
        # Get Object sizes
        self.n      = len(obj.data.vertices)
        self.n_edge = len(obj.data.edges) # sum([len(e) for e in obj.data.edges], 0)
        self.n_prim = len(obj.data.polygons) # (self.n - 1) * (self.n - 1) * 2

        # Get points a list of list (uneven list size)
        points = list()
        for v in obj.data.vertices:
            points.append([v.co.x, v.co.y, v.co.z])
        self.points = np.array(points, dtype=np.single)
        del[points]

        # Get edges as numpy
        neighbor_point = list()
        for i in range(self.n):
            neighbor_point.append(list())
        for e in obj.data.edges:
            p_id1 = e.vertices[0]
            p_id2 = e.vertices[1]
            neighbor_point[p_id2].append(p_id1)
            neighbor_point[p_id1].append(p_id2)
        self.neighbor_point = neighbor_point

        # Get primitives as numpy (either face or tetrahedron)
        # ...
        # ...

        # Init CPU fields
        self.initialize_fields()
        
        # Send to GPU fields
        ti.init(self._arch, unrolling_limit=0)
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.create_fields()

        self.isnot_init = False
        self.print_mesh_parameter()
        pass

    def create_fields(self):
        # Point variable
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.v = ti.Vector.field(3, dtype=float, shape=self.n)

        # Spring properties        
        self.l0 = ti.Vector.field(self.max_spring, dtype=float, shape=self.n)
        self.springs = ti.Vector.field(self.max_spring, dtype=int, shape=self.n)

        self.fill_points()
        self.fill_velocity()
        self.fill_springs()

    def initialize_fields(self):
        self.initialize_springs()

    def fill_points(self):
        self.x.from_numpy(self.points)
    
    @ti.kernel
    def fill_velocity(self):
        for i in self.v:
            self.v[i] = [0, 0, 0]
    
    def fill_springs(self):
        self.springs.from_numpy(self.edges)
        self.l0.from_numpy(np.array(self.all_l0, dtype=np.single))


    def initialize_springs(self):
        # Init springs/edges size (before fields !!!)
        # Create spring ids list    # and springs from objects (neighbor list and poitns)
        springs = list()
        for i, neib in enumerate(self.neighbor_point):
            s = neib.copy()
            neighbor_of_neighbors = sum([self.neighbor_point[p_id] for p_id in neib], [])
            if self.bending_springs:
                for np_id in neighbor_of_neighbors:
                    if np_id not in s+[i]:
                        s += [np_id]
            springs.append(s)
        # Fill edges to undefined value
        self.max_spring = max( [len(springs_list) for springs_list in springs] )
        for e in springs:
            diff = self.max_spring - len(e)
            if diff > 0:
                e += diff*[-1] 
        self.edges = np.array(springs, dtype=np.int32)

        # Create L0
        all_l0 = list()
        for i, neigh_list in enumerate(self.edges):
            xi = self.points[i]
            l0_list = list()
            for j in neigh_list:
                if j==-1: 
                    l0_list.append(0.0)
                    continue
                xj = self.points[j] # Write -1 if there is not j-th neighbor
                l0 = np.linalg.norm(xi-xj)
                l0_list.append(l0)
            all_l0.append(l0_list)
        self.all_l0 = all_l0

    def reset(self):
        self.fill_points()
        self.fill_velocity()

    @ti.kernel
    def step_forward(self):
        for i in self.x:
            x = self.x[i]
            v = self.v[i]
            l0 = self.l0[i]
            # Volumic Forces
            force = self.gravity# ti.Vector([gx, gy, gz]) # gravity

            # # Internal Forces
            for j in range(self.max_spring): # self.springs[i]:
                spring_id = self.springs[i][j]
                if spring_id==-1: 
                    continue
                spring_vec = self.x[spring_id] - x
                spring_vel = self.v[spring_id] - v
                dir = spring_vec.normalized()
                l = ti.math.length(spring_vec)

                f = self.spring_rigidity * (l-l0[j]) * dir
                f += spring_vel.dot(dir) * self.spring_damping * l*l
                force += self.spring_rigidity * (l-l0[j]) * dir
            v += force * self._dt
            v *= ti.exp(-self.air_drag * self._dt)

            # # Surfacic Forces (sphere collision)
            # distance_to_sphere_center = x - ti.Vector([0.0, 0.0, 0.0])
            # distance_to_sphere = distance_to_sphere_center.norm() - 1.0
            # if distance_to_sphere<=0:
            #     # Velocity projection
            #     normal = distance_to_sphere_center.normalized()
            #     v -= ti.min(v.dot(normal), 0) * normal

            x += self._dt * v / 2

            if x.z<0:
                x.x = x.x
                x.y = x.y
                x.z = -x.z
                v.x =  ti.sqrt(self.friction_coeff) * v.x
                v.y =  ti.sqrt(self.friction_coeff) * v.y
                v.z = -ti.sqrt(self.friction_coeff) * v.z
            self.x[i] = x
            self.v[i] = v
