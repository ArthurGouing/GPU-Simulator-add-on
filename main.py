#### Blender Import #### 
import bpy
from bpy.app.handlers import persistent
from bpy.utils import register_class, unregister_class
from mathutils import Vector

### Library Import ####
import sys, os
from time import perf_counter

dir = os.path.dirname(bpy.data.filepath)

if not dir in sys.path:
    sys.path.append(dir)
print(dir, bpy.data.filepath)
print(sys.path)

### Local Import ###
import mass_spring_simulator

# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(mass_spring_simulator)

from mass_spring_simulator import ExplicitMassSpringSimulator

cloth_sim = None

solver = [
    ("EMS", "Explicit Mass Spring", "Mass spring method to solve the dynamics", 0)
]
arch = [
    ("CPU", "Multi-CPU", "Multi CPU architecture backend to make the computation.", 0),
    ("GPU", "GPU", "Use the GPU to make the computation. The best backend will be automatically selected.", 1),
    ("VUL", "Vulkan", "Use the GPU with Vulkan backend to make the computation.", 2),
    ("METAL", "Metal", "Use the GPU with Metal backend to make the computation.", 3),
    ("CUDA", "Cuda", "Use the GPU with Cuda backend to make the computation.", 4)
]


# @persistent
def step_forward(scene):
    """
    Rules of computation according to frame change:
     - if we go to the next frame:     Compute one Simulation loop
     - if we go to the previous frame: Do nothing
     - if we jump to a father frame:   Reset the computation
     - if we compile and go to the next step: the computation is'nt init --> Faire l'init dans la compile ?
    """
    print("")
    print("---- Frame: ",scene.frame_current, "----")
    # Init local variables
    obj = scene.cloth_simulator.obj
    delta_frame = scene.frame_current - scene.frame_previous
    scene.frame_previous = scene.frame_current

    # Init
    if cloth_sim.isnot_init:
        t_init = perf_counter()
        cloth_sim.initialize_from_obj(scene.cloth_simulator.obj)
        print(f" Init time:   {(perf_counter()-t_init)*1e3:.3f} ms")

    # Compute 1 frame
    if delta_frame == 1:
        print("step formward")
        t_forward = perf_counter()
        cloth_sim.frame_forward()
        print(f"Forward time: {(perf_counter()-t_forward)*1e3:.3f} ms")

        t_update = perf_counter()
        cloth_sim.update_vertices(obj)
        print(f"Update time:  {(perf_counter()-t_update)*1e3:.3f} ms")

    # Reset Computation
    elif delta_frame != -1 and delta_frame != 0:
        # Do the pre-process part and reset  GPU buffers
        print("Reset the simulation")
        t_reset = perf_counter()
        cloth_sim.reset()
        print(f"Reset time:   {(perf_counter()-t_reset)*1e3:.3f} ms")
        t_update = perf_counter()
        cloth_sim.update_vertices(obj)
        print(f"Update time:  {(perf_counter()-t_update)*1e3:.3f} ms")


class ClothPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Cloth Simulator"
    bl_idname = "VIEW3D_PT_ClothSimu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GPU Simulation"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        sim = context.scene.cloth_simulator

        row = layout.row()
        row.label(text="Powered by Taichi", icon='MOD_CLOTH')

        if not cloth_sim:
            layout.prop(sim, "arch")
            layout.split()
            layout.prop(sim, "solver")
            layout.split()
            layout.operator("object.init_sim_operator", text="Init Simulator", icon="ADD")
        else:
            layout.label(text=f"Arch: {sim.arch}.")
            layout.label(text=f"The {sim.name} method is used")
            layout.operator("object.del_sim_operator", text="Delete Simulator", icon="PANEL_CLOSE")

        layout.label(text="Simulation parameters:")
        layout.prop(sim, "obj")
        layout.split()
        layout.prop(sim, "fps")
        layout.prop(sim, "dt")
        layout.split(factor=4.)
        layout.prop(sim, "mass")
        col = layout.column()
        col.prop(sim,'gravity')
        layout.split(factor=1.)
        if sim.solver=="EMS":
            layout.prop(sim, "k")
            layout.prop(sim, "alpha")
            layout.prop(sim, "drag")
            layout.split(factor=1.)
            layout.prop(sim, "bending")
        return

class InitSimulatorOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.init_sim_operator"
    bl_label = "Init Simulator Operator"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        sim_prop = context.scene.cloth_simulator
        global cloth_sim
        if sim_prop.solver=="EMS":
            cloth_sim=ExplicitMassSpringSimulator(sim_prop.arch)
        else:
            print(f"ERROR: the solver {sim_prop.solver} does not exist")
        return {'FINISHED'}
class DelSimulatorOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.del_sim_operator"
    bl_label = "Del Simulator Operator"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        global cloth_sim
        cloth_sim = None
        return {'FINISHED'}
        


def update_arch(self, context):
    cloth_sim.arch = self.arch
def update_fps(self, context):
    cloth_sim.fps = self.fps
def update_dt(self, context):
    cloth_sim.dt = self.dt
def update_gravity(self, context):
    cloth_sim.gravity[0] = self.gravity[0]
    cloth_sim.gravity[1] = self.gravity[1]
    cloth_sim.gravity[2] = self.gravity[2]
    cloth_sim.print_parameter()
def update_mass(self, context):
    cloth_sim.mass = self.mass
    cloth_sim.print_parameter()
def update_k(self, context):
    if isinstance(cloth_sim, ExplicitMassSpringSimulator):
        cloth_sim.spring_rigidity = self.k
    cloth_sim.print_parameter()
def update_alpha(self, context):
    if isinstance(cloth_sim, ExplicitMassSpringSimulator):
        cloth_sim.spring_damping = self.alpha
    cloth_sim.print_parameter()
def update_drag(self, context):
    if isinstance(cloth_sim, ExplicitMassSpringSimulator):
        cloth_sim.air_drag = self.drag
    cloth_sim.print_parameter()
def update_bending(self, context):
    if isinstance(cloth_sim, ExplicitMassSpringSimulator):
        cloth_sim.bending_springs = self.bending
    cloth_sim.print_parameter()

class ClothSimulationProperty(bpy.types.PropertyGroup):
    """
    Custom property structure.
    Belong to : bpy.types.Scene.cloth_simulator
    """
    arch:    bpy.props.EnumProperty(items=arch, name="Architecture", description="Specify the architecture and the backend that Majax simulator will use.", update=update_arch)
    solver:  bpy.props.EnumProperty(items=solver, name="Method", description="Choose the Solver method to use to make the simulation.")
    obj:     bpy.props.PointerProperty(type=bpy.types.Object, name="Cloth mesh", description="Object which will be simulated.")
    # Common to all solver parameters
    fps:     bpy.props.IntProperty(name="FPS", default=24, update=update_fps)
    dt:      bpy.props.FloatProperty(name="Delta Time", default=0.04, precision=6, description="Time between to sub simulation step. Smaller is the step, more stable is the simuation, but also slower.", update=update_dt)
    # Mass-spring parameters
    gravity: bpy.props.FloatVectorProperty(name="Gravity", subtype='ACCELERATION', default=Vector((0,0,9.81)), description="Direction and intensity of the gravity force, apply to all points.", update=update_gravity)
    mass:    bpy.props.FloatProperty(name="Mass", description="Mass totale of the object.")
    k:       bpy.props.FloatProperty(name="Spring rigidity", description="Rigidity of the springs. It represent the required quantity of force to move a points of 1kg to 1 meter.", update=update_k)
    alpha:   bpy.props.FloatProperty(name="Spring damping", description="A bigger damping, limite the speed movement of the cloth", update=update_alpha)
    drag:    bpy.props.FloatProperty(name="Air drag", description="Air drag applied on the cloth. A big air drag improve the stability.", update=update_drag)
    bending: bpy.props.BoolProperty(name="Bending springs", description="Use or no the Bending springs", update=update_bending)
    
    

classes = [InitSimulatorOperator, DelSimulatorOperator, ClothPanel]

def register():
    print("Add scene properties")
    ### Add Cloth properties to scene
    register_class(ClothSimulationProperty)
    bpy.types.Scene.cloth_simulator = bpy.props.PointerProperty(type=ClothSimulationProperty)
    bpy.types.Scene.frame_previous = bpy.props.IntProperty(name="p_frame", default=0)
    print("Register...")
    ### Register Class ###
    for cls in classes:
        print("register: ", cls.__name__)
        register_class(cls)
    print("Add Handlers")
    ### Add handlers
    bpy.app.handlers.frame_change_pre.append(step_forward)

def unregister():
    print("unregister")
    ### Delete handlers
    bpy.app.handlers.frame_change_pre.remove(
            bpy.app.handlers.frame_change_pre[0]
    )
    ### Unregister Classes ###
    for cls in reversed(classes):
        print(cls.__name__, cls)
        unregister_class(cls)
    ### Del Cloth properties to scene
    del bpy.types.scene.cloth_simulator
    del bpy.types.Scene.frame_previous


    
if __name__ == "__main__":
    # simulator = Simulator()
    try:
        unregister()
    except:
        print("unregister failed")
        pass
    register()
    print("Done loading script")