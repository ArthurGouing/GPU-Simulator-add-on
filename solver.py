###########################################################
# Description: 
#   Abstract class, used as an interface for all the method.
#   Provide the mandatory attribute and method to be
#   integrate within the Blender interface.
###########################################################

### Import
import bpy
import taichi as ti
import numpy as np
from abc import ABC, abstractmethod

class Solver(ABC):
    """
    Description:
    -------------
    General Abstract Class for physics solver


    Attributes:
    -------------
    name                 : The name of the solver
    arch                 : The architecture used by Taichi (GPU, mutli-CPU)
    precision            : Precision of the computation (single, double)
    dt                   : The time step of the simulation
    mass                 : The mass (should be solver specific)
    gravity              : The gravity values
    friction_coeff       : The friction coefficient value
    fps                  : The number of frame per seconds
    substep              : The number of iteration between 2 frames
    curr_time            : The time of the simulation
    is_not_init          : A boolean which indicate if the solver is initialised (i.e. Meshs are read)

    collider_freq        : The number of iteration where the solver look for collision 
    is_self_coll         : Tell if the solver use self collision method

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


    Methods:
    -------------
    initialize_from_obj    (...)
    frame_forward          (...)
    udpate_vertices        (...)
    udpate_collider_points (...)
    udpate_pin_position    (...)


    Abstract Methods:
    -------------
    get_field_size          (...)
    init_other_fields       (...)
    create_other_gpu_fields (...)
    step_forward            (...)
    find_collision          (...)
    reset                   (...)
    """

    def __init__(self, arch) -> None:
        # Architecture Computation
        self.name = "Undefined Solver"
        self.arch = arch # ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
        self.precision = np.single

        # Simulation Parameters 
        self._dt = 0.04
        self._fps = 24
        self._substeps = int(1 / self._fps // self._dt)
        self.curr_time =  0
        self.collider_freq = 10 # self._substeps
        self.is_self_coll = False

        self.mass = 1.
        self._mass = 1.
        self.gravity = ti.Vector([0, 0, -9.81])
        self.friction_coeff = 0.70

        self.isnot_init = True

    @property
    def arch(self):
        if self._arch==ti.cpu:
            return "Multi-CPU"
        if self._arch==ti.gpu:
            return "GPU"
        elif self._arch==ti.vulkan:
            return "GPU Vulkan"
        elif self._arch==ti.metal:
            return "GPU Metal"
        elif self._arch==ti.cuda:
            return "GPU Cuda"
        else:
            return self._arch
    @arch.setter
    def arch(self, arch_str):
        """ apparamment amdgpu, dx12, opengl, gles, existent aussi ..."""
        if arch_str=='GPU':
            self._arch = ti.gpu
        elif arch_str=='CPU':
            self._arch = ti.cpu
        elif arch_str=='VULKAN':
            self._arch = ti.vulkan
        elif arch_str=='METAL':
            self._arch = ti.metal
        elif arch_str=='CUDA':
            self._arch = ti.cuda
        else:
            print("ERROR: this arch doesn't exist")

    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self, new_dt):
        self._dt = new_dt
        self._substeps = int(1 / self._fps // self._dt)
        print("substeps=", self._substeps)

    @property
    def fps(self):
        return self._fps
    @fps.setter
    def fps(self, new_fps):
        self._fps = new_fps
        self._substeps = int(1 / self._fps // self._dt)
        print("substeps=", self._substeps)

    def print_parameter(self):
        # Print Simulation Informations
        print("")
        print(f"  Simulating with {self.name} method")
        print("")
        print("--Simulation parameters-----------------------")
        print(f" {'arch':<10}: {self.arch}")
        print(f" {'dt':<10}: {self._dt}")
        print(f" {'fps':<10}: {self._fps}")
        print(f" {'substeps':<10}: {self._substeps}")
        print("")
        print(f" {'mass':<10}: {self.mass}")
        print(f" {'gravity':<10}: {self.gravity}")

    def initialize_from_obj(self, 
            obj: bpy.types.Object, 
            collider: bpy.types.Object=None, 
            animated_collider: bpy.types.Object=None, 
            pin_group_id: int=None, 
            animated_pin_group_id: int=None
        ):
        """
        Read the Bpy object for the simulated object, the collider, and the pin group.
        And store all the data in np.array, then send the data to the GPU.
        """
        # Get Objects size
        self.n      = len(obj.data.vertices)
        self.n_edge = len(obj.data.edges) # sum([len(e) for e in obj.data.edges], 0)
        self.n_prim = len(obj.data.polygons) # (self.n - 1) * (self.n - 1) * 2
        self.n_collider      = len(collider.data.vertices) if collider else 0
        self.n_anim_collider = len(animated_collider.data.vertices) if animated_collider else 0
        if collider or animated_collider:
            self.is_coll = True
        else:
            self.is_coll = False # inutile, car déjà mis à False dans l'init
        self.is_pin = False
        self.get_field_size()
        # Should be doing self.get_field_size 
        # and in the daugther class :
        # super.get_field_size(...)
        # self.get_field_size(...)
        # TODO rewrite

        # Get points
        # Get pin and animated_pin index
        points = list()
        animated_pin = list()
        pin = list()
        for v in obj.data.vertices:
            points.append([v.co.x, v.co.y, v.co.z])
            for g in v.groups:
                if g==animated_pin_group_id:
                    animated_pin.append(v.index)
                elif g==pin_group_id:
                    pin.append(v.index)
        self.points = np.array(points, dtype=self.precision)
        self.pin = pin # np.array(pin, dtype=self.precision)
        self.animated_pin = animated_pin # np.array(animated_pin, dtype=self.precision)
        del[points]
        del[pin]
        del[animated_pin]

        # Get edges as numpy array
        neighbor_point = list()
        for i in range(self.n):
            neighbor_point.append(list())
        for e in obj.data.edges:
            p_id1 = e.vertices[0]
            p_id2 = e.vertices[1]
            neighbor_point[p_id2].append(p_id1)
            neighbor_point[p_id1].append(p_id2)
        self.neighbor_point = neighbor_point
        del neighbor_point

        # Get collider points
        if collider:
            collider_points = list()
            for v in collider.data.vertices:
                collider_points.append([v.co.x, v.co.y, v.co.z])
            self.collider_points      = np.array(collider_points, dtype=self.precision)
            del[collider_points]
        else:
            self.collider_points = None

        # Get animated collider points
        if animated_collider:
            anim_collider_points = list()
            for v in animated_collider.data.vertices:
                anim_collider_points.append([v.co.x, v.co.y, v.co.z])
            self.anim_collider_points = np.array(anim_collider_points, dtype=self.precision)
            del[anim_collider_points]
        else:
            self.anim_collider_points = None

        # Other field initialisation for daughter class
        self.init_other_field(obj)

        
        # Send to GPU fields
        ti.init(self._arch)
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.v = ti.Vector.field(3, dtype=float, shape=self.n)

        if self.is_coll:
            self.x_collider = ti.Vector.field(3, dtype=float, shape=self.n_collider) # need to add anim_collider to this x_collider Taichi array

        # self.x_pin = ti.Vector.field(3, dtype=self.precision, shape=self.n_anim_pin)
        # self.id_pin = ti.field(int, shape=self.n_anim_pin+self.n_pin)

        # Other field GPU send for daughter class
        self.create_other_gpu_fields()

        self.isnot_init = False
        self.print_mesh_parameter()
        pass

    @abstractmethod
    def get_field_size(self):
        pass
    @abstractmethod
    def init_other_field(self, obj:bpy.types.Object):
        pass
    @abstractmethod
    def create_other_gpu_fields(self):
        pass

    def print_mesh_parameter(self):
        print("")
        print("--Mesh parameters-----------------------")
        print(f" {'n':<10}: {self.n}")
        print(f" {'n edges':<10}: {self.n_edge}")
        print(f" {'n prim':<10}: {self.n_prim}")
        print(f" {'memory':<10}: {'TODO'}")

    def update_vertices(self, obj: bpy.types.Object):
        import array
        vert = obj.data.vertices
        points_array = self.x.to_numpy().ravel().tolist()
        seq = array.array('f', points_array)
        vert.foreach_set('co', seq)
        obj.data.update()
    
    @abstractmethod
    def reset(self):
        pass
        # curr_time = 0
        # self.initialize_point()
        # self.initialize_velocity()

    def frame_forward(self, animated_collider: bpy.types.Object, pin: np.array):
        # Update interaction points
        if animated_collider:
            print("update collider")
            self.update_collider_points(animated_collider) # à optimiser à la manière de self.update_vertices
        if self.is_pin: # TODO replace by animated_pin
            print("udpate pin")
            self.update_pin_position(pin)

        for t in range(self._substeps):
            if self.is_coll and t%20==0:
                print("find_collisision", self.collider_freq, self.curr_time%self.collider_freq)
                self.find_collision()

            print("stepforward 2")
            self.step_forward()# self.gravity[0], self.gravity[1], self.gravity[2])
            self.curr_time += 1
        print(f" {'Rcollision':<10}: {self.r_coll}")

    def update_collider_points(self, collider:bpy.types.Object):
        import array
        vert = collider.data.vertices
        points_array = self.anim_collider_points# self.x.to_numpy().ravel().tolist()
        seq = array.array('f', points_array)
        vert.foreach_get('co', seq)
        self.x_collider.from_numpy(self.anim_collider_points)

    def update_pin_position(self, pin_points: np.array):
        self.x_pin.from_numpy(pin_points)
        self.fill_pin()
        self.kernel_up_pin()

    @ti.kernel
    def kernel_up_pin(self):
        for i in self.x_pin:
            id = self.pin_id[i]
            self.x[id] = self.x_pin[i]

    @abstractmethod
    def step_forward():
        pass

    @abstractmethod
    def find_collision():
        pass