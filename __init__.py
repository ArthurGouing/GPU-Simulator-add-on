#### Blender Import ####
print("Start init")
import bpy
from bpy.app.handlers import persistent # To keep handlers on new file opening
from bpy.utils import register_class, unregister_class
from mathutils import Vector

### Local Library Import ####
print("import local lib")
import sys, os
from time import perf_counter

# If script debug :
# dir = os.path.dirname(bpy.data.filepath)
# 
# print(dir, bpy.data.filepath)
# print(sys.path)
# Else publish extension:
extension_directory = bpy.utils.extension_path_user(__package__, path="", create=True)
print("dir:", extension_directory)
# if not extension_directory in sys.path:
#     sys.path.append(extension_directory)


### Local Import ###
from . import solver
from . import exp_mass_spring
from . import xpbd

# this next part forces a reload in case you edit the source after you first start the blender session
# If script debug:
# import imp
# imp.reload(solver)
# imp.reload(exp_mass_spring)
# imp.reload(xpbd)

from .exp_mass_spring import ExplicitMassSpring
from .xpbd import PositionBasedDynamic

print("import taichi")
import taichi
print("end import")

# Global variable which contoain the simulator Object
cloth_sim = None

solver_item = [
    ("EMS", "Explicit Mass Spring", "Mass spring method to solve the dynamics", 0),
    ("XPBD", "Position Based Dynamics", "Use the well known XPBD solver. This solver is more stable, specially for stiff material", 1)
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
     - if we jump to a further previous frame:   Reset the computation
     - if we compile and go to the next step: Reset the computation (the computation is'nt init --> Faire l'init dans la compile ?)
    """
    print("")
    print("---- Frame: ",scene.frame_current, "----")
    # Init local variables
    obj = scene.cloth_simulator.obj
    anim_collider = None # scene.cloth_simulator.animated_collider # TODO
    anim_pin = scene.cloth_simulator.animated_pin # is a string ("Pin_Group_name" or None)
    delta_frame = scene.frame_current - scene.frame_previous
    scene.frame_previous = scene.frame_current
    # Init
    if cloth_sim.isnot_init:
        t_init = perf_counter()
        obj = scene.cloth_simulator.obj # Useless
        collider = scene.cloth_simulator.collider
        # anim_collider = scene.cloth_simulator.animated_collider
        anim_collider = None
        # id_pin = scene.cloth_simulator.pin.index
        id_pin = None
        id_anim_pin = scene.cloth_simulator.animated_pin.index
        cloth_sim.initialize_from_obj(obj, collider=collider, 
            animated_collider=anim_collider, animated_pin_group_id=id_anim_pin, 
            pin_group_id=id_pin
        )
        print(f" Init time:   {(perf_counter()-t_init)*1e3:.3f} ms")

    # Compute 1 frame
    if delta_frame == 1:
        print("step formward")
        t_forward = perf_counter()
        cloth_sim.frame_forward(anim_collider, anim_pin)
        print(f"Forward time: {(perf_counter()-t_forward)*1e3:.3f} ms")

        t_update = perf_counter()
        cloth_sim.update_vertices(obj) # cloth_sim.update_objects ou juste update # TODO:rename
        print(f"Update time:  {(perf_counter()-t_update)*1e3:.3f} ms")

    # Reset Computation
    elif delta_frame != -1 and delta_frame != 0:
        # Do the pre-process part and reset  GPU buffers
        print("Reset the simulation")
        t_reset = perf_counter()
        cloth_sim.reset()
        print(f"Reset time:   {(perf_counter()-t_reset)*1e3:.3f} ms")
        t_update = perf_counter()
        cloth_sim.update_vertices(obj) # Idem
        print(f"Update time:  {(perf_counter()-t_update)*1e3:.3f} ms")


class ClothBasePanel(bpy.types.Panel):
    """Creates a Panel Allowing to specify all the parameters related to the Cloth simulatioon in the Object properties window"""
    bl_label = "Method"
    # bl_idname = "VIEW3D_PT_ClothSimu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GPU Simulation"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        sim_obj = context.scene.cloth_simulator.obj
        sim = context.scene.cloth_simulator

        layout.label(text="Select the solver", icon='MEMORY')
        if not cloth_sim:
            layout.prop(sim, "arch")
            layout.split()
            layout.prop(sim, "solver")
            layout.split()
            layout.operator("object.init_sim_operator", text="Create Simulator", icon="ADD")
        else:
            layout.label(text=f"Arch: {sim.arch}.")
            layout.label(text=f"The {sim.name} method is used")
            layout.operator("object.del_sim_operator", text="Delete Simulator", icon="PANEL_CLOSE")
        return
class ClothInitPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Initialisation"
    # bl_idname = "VIEW3D_PT_ClothSimu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GPU Simulation"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        obj = context.object
        sim_obj = context.scene.cloth_simulator.obj
        sim = context.scene.cloth_simulator
        flow = layout

        flow.label(text="Link Objects with the simulator:", icon='LINK_BLEND')
        flow.label(text="Simulated object:")
        col = layout.column()# align=False, heading="Simulated object")
        col.prop(sim, "obj")
        col.prop(sim, "fps")
        col.prop(sim, "dt")
        layout.split()
        flow.label(text="Colliding object:")
        col = flow.column()
        col.prop(sim, "collider")
        col.prop(sim, "animated_collider")
        col.prop(sim, "coll_freq")
        col.prop(sim, "is_self_coll")
        layout.split()
        flow.label(text="Pinnded groups:")
        # layout.prop(sim, "animated_pin")
        col = flow.column()
        col.prop_search(sim, "pin", sim_obj, "vertex_groups", text="Pins")
        col.prop_search(sim, "animated_pin", sim_obj, "vertex_groups", text="Animated Pins")
        return
class ClothParamPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Parameters"
    # bl_idname = "VIEW3D_PT_ClothSimu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GPU Simulation"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        obj = context.object
        sim_obj = context.scene.cloth_simulator.obj
        sim = context.scene.cloth_simulator
        flow = layout

        layout.split(factor=4.)
        layout.prop(sim, "mass")
        col = layout.column()
        col.prop(sim,'gravity')
        layout.split(factor=1.)
        if sim.solver=="EMS":
            col = layout.column()
            col.prop(sim, "k")
            col.prop(sim, "k_bend")
            col.prop(sim, "alpha")
            col.prop(sim, "drag")
        if sim.solver=="XPBD":
            col = layout.column()
            layout.prop(sim, "compliance")
            layout.prop(sim, "bending_compliance")
            layout.prop(sim, "stretch_relaxation")
            layout.prop(sim, "bending_compliance")
            layout.prop(sim, "stretch_relaxation")
            layout.prop(sim, "bending_relaxation")
            layout.prop(sim, "friction")
            layout.split(factor=1.)
            layout.prop(sim, "bending")
            layout.prop(sim, "ground")
        return

class InitSimulatorOperator(bpy.types.Operator):
    """Operator which initialize the Cloth_sim Object from the simulator class"""
    bl_idname = "object.init_sim_operator"
    bl_label = "Init Simulator Operator"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        sim_prop = context.scene.cloth_simulator
        global cloth_sim
        if sim_prop.solver=="EMS":
            cloth_sim=ExplicitMassSpring(sim_prop.arch)
        elif sim_prop.solver=="XPBD":
            cloth_sim=PositionBasedDynamic(sim_prop.arch)
        else:
            print(f"ERROR: the solver {sim_prop.solver} does not exist")

        update_sim_prop(sim_prop, context)

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
        context.scene.frame_set(0)
        cloth_sim = None
        return {'FINISHED'}

# TODO move to the InitSimulatorOperator class
def update_sim_prop(self, context):
    # Base
    cloth_sim.arch = self.arch
    cloth_sim.fps = self.fps
    cloth_sim.dt = self.dt
    # Common
    cloth_sim.gravity[0] = self.gravity[0]
    cloth_sim.gravity[1] = self.gravity[1]
    cloth_sim.gravity[2] = self.gravity[2]
    cloth_sim.mass = self.mass
    # Interaction
    cloth_sim.is_collider = self.collider
    cloth_sim.is_anim_collider = self.animated_collider
    cloth_sim.collider_freq = self.coll_freq
    cloth_sim.is_pin = self.pin
    cloth_sim.is_anim_pin = self.animated_pin
    cloth_sim.is_self_coll = self.is_self_coll
    # Parameters
    if isinstance(cloth_sim, ExplicitMassSpring):
        cloth_sim.spring_rigidity = self.k
        cloth_sim.bend_rigidity = self.k_bend
        cloth_sim.spring_damping = self.alpha
        cloth_sim.air_drag = self.drag
        cloth_sim.bending_springs = self.is_bend
    if isinstance(cloth_sim, PositionBasedDynamic):
        cloth_sim.is_bending = self.bending
        cloth_sim.compliance = self.compliance
        cloth_sim._bcompliance = self.bending_compliance
        cloth_sim.relax_coeff_stretch = self.stretch_relaxation
        cloth_sim.relax_coeff_bending = self.bending_relaxation
        cloth_sim.friction_coeff = self.friction
        cloth_sim.bending_springs = self.bending_compliance
        cloth_sim.ground = self.ground
    cloth_sim.print_parameter()

class ClothSimulationProperty(bpy.types.PropertyGroup):
    """
    Custom property structure.
    Belong to : bpy.types.Scene.cloth_simulator
    """
    arch:    bpy.props.EnumProperty(items=arch, name="Architecture", description="Specify the architecture and the backend that Majax simulator will use.", update=update_sim_prop)
    solver:  bpy.props.EnumProperty(items=solver_item, name="Method", description="Choose the Solver method to use to make the simulation.")
    obj:     bpy.props.PointerProperty(type=bpy.types.Object, name="Cloth mesh", description="Object which will be simulated.")
    # Interactor
    animated_collider: bpy.props.PointerProperty(type=bpy.types.Object, name="Animated Collider", description="Object which will act on the simulation.")
    collider:          bpy.props.PointerProperty(type=bpy.types.Object, name="Collider", description="Object which will act on the simulation which can be animated.")
    animated_pin:      bpy.props.StringProperty(name="Animated Pin", description="Points which aren't affected by the simulation and are update each frames")
    pin:               bpy.props.StringProperty(name="Pin", description="Points which aren't affected by the simulation")
    is_self_coll:      bpy.props.BoolProperty(name="Self collision", description="Enable self collision computation if checked")
    # Common to all solver parameters
    fps:        bpy.props.IntProperty(name="FPS", default=24, update=update_sim_prop)
    dt:         bpy.props.FloatProperty(name="Delta Time", default=0.004, precision=6, description="Time between to sub simulation step. Smaller is the step, more stable is the simuation, but also slower.", update=update_sim_prop)
    gravity:    bpy.props.FloatVectorProperty(name="Gravity", subtype='ACCELERATION', default=Vector((0,0,9.81)), description="Direction and intensity of the gravity force, apply to all points.", update=update_sim_prop)
    mass:       bpy.props.FloatProperty(name="Mass", description="Mass totale of the object.", default=1.)
    mass_field: bpy.props.StringProperty(name="Field Mass", description="Mass value for each point")
    vel_field:  bpy.props.StringProperty(name="Field Velocity", description="Initial value of the velocity for each point")
    friction:   bpy.props.FloatProperty(name="Friction", description="Friction of the particles when colliding. It correspond to the % energy preserved at each iteration.", update=update_sim_prop, default=0.9)
    coll_freq:  bpy.props.FloatProperty(name="Collision frequency", description="Time spread where collision detection computation occurs. ", update=update_sim_prop, default=24)
    # Mass-spring parameters
    k:       bpy.props.FloatProperty(name="Spring rigidity", description="Rigidity of the springs. It represent the required quantity of force to move a points of 1kg to 1 meter.", update=update_sim_prop, default=1e4)
    k_bend:  bpy.props.FloatProperty(name="Bending rigidity", description="Rigidity of the springs.", update=update_sim_prop)
    is_bend: bpy.props.BoolProperty(name="Bending springs", description="Use or no the Bending springs", update=update_sim_prop)
    alpha:   bpy.props.FloatProperty(name="Spring damping", description="A bigger damping, limite the speed movement of the cloth", update=update_sim_prop, default=1e4)
    drag:    bpy.props.FloatProperty(name="Air drag", description="Air drag applied on the cloth. A big air drag improve the stability.", update=update_sim_prop, default=1.)
    # XPBD parameters
    compliance: bpy.props.FloatProperty(name="Compliance", description="Caracterize the flexibility of the constraint. A smaller value generate stiff constraint and more rigid objects dynamics", update=update_sim_prop, default=0.)
    bending_compliance: bpy.props.FloatProperty(name="Bending Compliance", description="Caracterize the flexibility of the constraint. A smaller value generate stiff constraint and more rigid objects dynamics", update=update_sim_prop, default=0.)
    stretch_relaxation: bpy.props.FloatProperty(name="Stretch Relaxation", description="divide the dX update by the relaxation coefficient. A small value impose a slower evoluiton of the system which improve the stability.", update=update_sim_prop, default=1.)
    bending_relaxation: bpy.props.FloatProperty(name="Bending Relaxation", description="divide the dX update by the relaxation coefficient. A small value impose a slower evoluiton of the system which improve the stability.", update=update_sim_prop, default=1.)
    ground:     bpy.props.BoolProperty(name="Ground", description="Use a ground which the object collide with.", update=update_sim_prop)


classes = [InitSimulatorOperator, DelSimulatorOperator, ClothBasePanel, ClothInitPanel, ClothParamPanel]


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
    try:
        unregister()
    except:
        print("unregister failed")
        pass
    register()
    print("Done loading script")
