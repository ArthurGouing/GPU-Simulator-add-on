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

    Attributes:
    -------------

    Methods:
    -------------
    """

    def __init__(self, arch) -> None:
        # Architecture Computation
        self.name = "Undefined Solver"
        self.arch = arch # ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda

        # Simulation Parameters 
        self._dt = 0.04
        self._fps = 24
        self._substeps = int(1 / self._fps // self._dt)
        self.curr_time =  0

        self.mass = 1.
        self._mass = 1.
        self.gravity = ti.Vector([0, 0, -9.81])

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
        # self.initialize_fields()
        
        # Send to GPU fields
        ti.init(self._arch)
        self.x = ti.Vector.field(3, dtype=float, shape=self.n)
        self.create_fields()

        self.isnot_init = False
        self.print_mesh_parameter()
        pass

    @abstractmethod
    def create_fields(self):
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
        # self.initialize_point()
        # self.initialize_velocity()

    def frame_forward(self):
        for t in range(self._substeps):
            self.step_forward()# self.gravity[0], self.gravity[1], self.gravity[2])
            self.curr_time += t

    @abstractmethod
    def step_forward():
        pass