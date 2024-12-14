import bpy
import taichi as ti
import taichi.math as mti
import numpy as np

from solver import Solver

@ti.data_oriented
class ExplicitMassSpring(Solver):
    """
    Description: 
    -------------
    Solver for the basic expicit mass spring method 


    Inherited Attributes:
    -------------
    dt                   : The time step of the simulation
    mass                 : The mass (should be solver specific)
    gravity              : The gravity values
    friction_coeff       : The friction coefficient value
    fps                  : The number of frame per seconds
    substep              : The number of iteration between 2 frames
    n                    : The number of verticies in the simulated object
    n_edge               : The number of edges     in the simulated object
    n_prim               : The number of polygons  in the simulated object
    n_collider           : The number of verticies in the static collider object
    n_anim_collider      : The number of verticies in the anumated collider object
    is_coll              : Indicate if there is collider in the simulation
    is_pin               : Indicate if there is pinned vertices in the simulation object
    points               : Numby array of the verticies of the simulated object
    pin                  : Python list of the static pinned vertices
    animated_pin         : Python list of the animated pinned vertices
    neighbor_point       : python list of the list of neibhot for each vertices (size=n, n_neighb)
    collider_points      : Numpy array of the verticies of the static collider object
    anim_collider_points : Numpy array of the verticies of the animated collider object
    x                    : Taichi array that represent the vertices position
    v                    : Taichi array that represent the vertices velocity
    x_collider           : Taichi array that represent the collider vertices position

    Attributes:
    -------------
    spring_rigidity      : Parameter which indicate how much force we need to stretch an edge
    bend_rigidity        : Parameter which indicate how much force we need to bend an edge
    spring_damping       : Parameter which indicate the spring damping
    air_drag             : Parameter which indicate the air drag apply to the vertices

    l0                   : Taichi array (n, max_spring) which contains the distance between all the neighbor for each vertices
    springs              : Taichi array (n, max_spring) which contains the id of all the neighbor for each vertices
    collider             : Taichi struct which contain the data required to compute the collider response force (id, dist and contact plan normal)



    Methods:
    -------------
    get_field_size          (...)
    init_other_fields       (...)
    create_other_gpu_fields (...)
    step_forward            (...)
    find_collision          (...)
    reset                   (...)
    """

    def __init__(self, arch) -> None:
        super().__init__(arch)
        self.name = "Explicit Mass Spring"

        # Spring properties
        self.spring_rigidity = 1.7e5
        self.bend_rigidity   = 1.7e5
        self.spring_damping  = 1e9
        self.air_drag = 8 # 1.5


        self.bending_springs = True

    def print_parameter(self):
        super().print_parameter()
        print(f" {'rigidity':<10}: {self.spring_rigidity}")
        print(f" {'damping':<10}: {self.spring_damping}")
        print(f" {'drag':<10}: {self.air_drag}")
        print(f" {'Bending':<10}: {self.bending_springs}")
        print("------------------------------------------------------")
        return

    def get_field_size(self):
        pass

    def init_other_field(self, obj):
        self.initialize_springs()
        pass

    def create_other_gpu_fields(self):
        # Spring properties        
        self.l0 = ti.Vector.field(self.max_spring, dtype=float, shape=self.n)
        self.springs = ti.Vector.field(self.max_spring, dtype=int, shape=self.n)

        # Collider info
        if self.is_coll:
            self.collider = ti.Struct.field({"id": int, "dist": float, "normal": mti.vec3}, shape=self.n)

        # self.x.from_numpy(self.points)
        self.fill_points()
        self.fill_collider_points()
        self.fill_velocity()
        self.fill_springs()


    def fill_points(self):
        self.x.from_numpy(self.points)

    def fill_collider_points(self):
        self.x.from_numpy(self.collider_points)
    
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

        self.r_coll = 0.9 * min([v for v_l0 in all_l0 for v in v_l0]) / 2

        del[all_l0]

    def reset(self):
        self.fill_points()
        self.fill_velocity()

    @ti.kernel
    def find_collision(self):
        """"
        Use Sphere collision detection
        """
        print("Start find_collision")
        for i in self.x:
            for j in range(self.n_collider):
                vec = self.x[i]-self.x_collider[j]
                dist = mti.length(vec)
                if  dist < 2*self.r_coll:
                    self.collider[i].id = j
                    self.collider[i].dist = dist
                    self.collider[i].normal = vec / dist
                else:
                    self.collider[i].id = -1
                print(self.collider[i].id)
        pass

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
            if self.is_coll:
                collider_pid = self.collider[i].id
                if collider_pid > -1:
                    vec = x - self.x_collider[collider_pid]
                    dist = mti.length(vec) - self.r_coll
                    if dist <= 0:
                        # Velocity projection
                        # set velocity component normal to contact surface equal to 0
                        normal = vec.normalized()
                        v -= ti.min(v.dot(normal), 0) * normal
                        # Add forces equivakent to collisiont
                        # v = dist**(3/2) * normal
                        # Or add a energy convertion term
                        # v -= (1+0.9) * ti.min(v.dot(normal), 0) * normal

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
