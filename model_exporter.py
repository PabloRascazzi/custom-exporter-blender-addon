import bpy
import sys
import numpy
from enum import Enum

bl_info = {
    "name": "Model Exporter Add-on",
    "description": "Exports model data with vertices and indices in a proper format for OpenGL.",
    "author": "Pablo Rascazzi",
    "version": (0, 5),
    "blender": (2, 92, 0),
    "location": "File > Export > Model Exporter (.model)",
    "category": "Import-Export"
}

ra = 6 # Rounding Accuracy

class Error(Exception):
    pass


class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class ModelType(OrderedEnum):
    xyz            = 0
    xyzuv          = 1
    xyztbn         = 2
    xyzuvtbn       = 3
    xyzRigged      = 4
    xyzuvRigged    = 5
    xyztbnRigged   = 6
    xyzuvtbnRigged = 7


class Model():
    def __init__(self, type, vertex_stride, vertex_count, vertices, material_count, indices_list):
        self.type = type
        self.vertex_stride = vertex_stride
        self.vertex_count = vertex_count
        self.vertices = vertices
        self.material_count = material_count
        self.indices_list = indices_list # A list of indices array
    
    def log(self):
        print(' > Model Type: %s' % self.type.name)
        print(' == > Vertex Stride: %d' % self.vertex_stride)
        print(' == > Vertex Count: %d' % self.vertex_count)
        print(' == > Vertices Array:', self.vertices)
        print(' == > Material Count: %d' % self.material_count)
        for i in range(self.material_count):
            print(' == == > Indices[%d] Size: %d' % (i, len(self.indices_list[i])))
            print(' == == > Indices[%d] Array:' % i, self.indices_list[i])
        

class BoneInfo():
    def __init__(self):
        self.index = None
        self.name = None
        self.local_pose_transform = None
        self.children_indices = []
        
    def log(self):
        print(' > Bone Name: %s' % self.name)
        print(' == > Index: %d' % self.index)
        print(' == > Children Indices:', self.children_indices)
        print(' == > Matrix:', self.local_pose_transform[0:4])
        print('             ', self.local_pose_transform[4:8])
        print('             ', self.local_pose_transform[8:12])
        print('             ', self.local_pose_transform[12:16])


def matrix_to_float16(matrix):
    float16 = []
    float16.extend([round(matrix[0][0], ra), round(matrix[0][1], ra), round(matrix[0][2], ra), round(matrix[0][3], ra)])
    float16.extend([round(matrix[1][0], ra), round(matrix[1][1], ra), round(matrix[1][2], ra), round(matrix[1][3], ra)])
    float16.extend([round(matrix[2][0], ra), round(matrix[2][1], ra), round(matrix[2][2], ra), round(matrix[2][3], ra)])
    float16.extend([round(matrix[3][0], ra), round(matrix[3][1], ra), round(matrix[3][2], ra), round(matrix[3][3], ra)])
    return float16


def process_armature_data(armature, logging):
    if logging == True: print("Processing armature data...")   
    bones_dict = {} # dict of bones with bone names as key (bone.name == vertex_group.name)
    
    # Process BoneInfo from armature data
    next_index = 0
    for bone in armature.bones:   
        new_bone = BoneInfo()
        new_bone.index = next_index
        new_bone.name = bone.name
        if bone.parent == None:
            new_bone.local_pose_transform = matrix_to_float16(bone.matrix_local)
        else:
            new_bone.local_pose_transform = matrix_to_float16(bone.parent.matrix_local.inverted() * bone.matrix_local)
        bones_dict[bone.name] = new_bone
        next_index += 1
    
    # Process BoneInfo Children Indices List
    for bone in armature.bones:
        for children in bone.children:
            bones_dict[bone.name].children_indices.append(bones_dict[children.name].index)

    # Log BoneInfo dictionary
    if logging == True:  
        for key in bones_dict: bones_dict[key].log()
            
    return bones_dict


def process_model_data(object, model_type, bones_dict, logging):
    if logging == True: print("Processing model data...")
    
    # Initialize variables and data
    mesh = object.data
    loop = mesh.loops
    vertex_group = object.vertex_groups
    vertex_stride = 3
    bones_per_vertex = 3
    
    if 'uv' in model_type.name: 
        vertex_stride += 2
        uv_layer = mesh.uv_layers.active.data
    if 'tbn' in model_type.name: 
        vertex_stride += 3
        mesh.calc_tangents()
    if 'Rigged' in model_type.name:
        vertex_stride += (bones_per_vertex*2)
    
    vert_dict = {}
    vertices = []
    indices = [[]]
    
    next_index = 0
    for face in mesh.polygons:
        tmp_indices = []

        # processing vertices per faces...
        for li in face.loop_indices:
            vi = loop[li].vertex_index # vertex Index
            vert = []
            vert.append(round(mesh.vertices[vi].co.x, ra))
            vert.append(round(mesh.vertices[vi].co.z, ra))
            vert.append(round(-mesh.vertices[vi].co.y, ra))
            if 'uv' in model_type.name:
                vert.append(round(uv_layer[li].uv.x, ra))
                vert.append(round(uv_layer[li].uv.y, ra))
            if 'tbn' in model_type.name:
                vert.append(round(loop[li].normal[0], ra))
                vert.append(round(loop[li].normal[1], ra))
                vert.append(round(loop[li].normal[2], ra))
            if 'Rigged' in model_type.name:
                # WORK IN PROGRESS              
                vb = [] # vertex bones
                for group in mesh.vertices[vi].groups:
                    vb.append([bones_dict[vertex_group[group.group].name].index, group.weight])
                
                vb = sorted(vb, reverse=True)[:bones_per_vertex]
                if len(vb) < bones_per_vertex: # fill bone array with [-1,0] if too short
                    vb.extend([[-1,0]] * (bones_per_vertex-len(vb)))
                
                # Normalize all bone weights
                total_weight = 0
                for bi in range(bones_per_vertex): total_weight += vb[bi][1]
                for bi in range(bones_per_vertex): vb[bi][1] = vb[bi][1]/total_weight
                  
                # Add bone indices and bone weights to vertex
                for bi in range(bones_per_vertex): vert.append(int(vb[bi][0]))
                for bi in range(bones_per_vertex): vert.append(round(vb[bi][1], ra))
            
            vert_key = str(vert)
            if vert_key not in vert_dict:
                vert_dict[vert_key] = next_index
                vertices.extend(vert)
                tmp_indices.append(next_index)
                next_index += 1
            else:
                tmp_indices.append(vert_dict[vert_key])
        
        # processing indices per faces...
        if len(tmp_indices) < 3:
            print('Warning: Points and lines currently not supported.')   
        else:
            if len(indices[face.material_index]) > 0: indices[face.material_index].append(-1)
                
            if len(tmp_indices) == 3:
                indices[face.material_index].append(tmp_indices[0])
                indices[face.material_index].append(tmp_indices[1])
                indices[face.material_index].append(tmp_indices[2])
            elif len(tmp_indices) == 4: 
                indices[face.material_index].append(tmp_indices[0])
                indices[face.material_index].append(tmp_indices[1])
                indices[face.material_index].append(tmp_indices[3])
                indices[face.material_index].append(tmp_indices[2])
            elif len(tmp_indices) > 4:
                for i in range(len(tmp_indices)):
                    indices[face.material_index].append(tmp_indices[0])
                    indices[face.material_index].append(tmp_indices[i])

    new_model = Model(model_type, vertex_stride, len(vert_dict), vertices, len(indices), indices)
    return new_model


def make_objects_list(selection, logging):
    if logging == True: print('Fetching objects list...')
    mesh_list = []
    armature_list = []
   
    def recursive_search(object, recursive = True):
        if object.type == 'MESH':
            if object not in mesh_list: 
                mesh_list.append(object)
        for modifier in object.modifiers:
            if modifier.type == 'ARMATURE':
                if modifier.object not in armature_list:
                    armature_list.append(modifier.object)
                break
        if recursive == True:
            for child in object.children:
                if child == None: continue
                recursive_search(child)
            
    
    recursive = False
    selected_objects = []     
    if selection == 'selected_only':
        recursive = False
        selected_objects = bpy.context.selected_objects
    elif selection == 'selected_children':
        recursive = True
        selected_objects = bpy.context.selected_objects
    elif selection == 'all_objects':
        recursive = True
        selected_objects = bpy.context.visible_objects
    else: raise Error('Invalid selection.')

    for object in selected_objects:
        if object == None:  continue 
        recursive_search(object, recursive)
    
    if len(mesh_list) == 0: 
        raise Error('Object selection(s) have no mesh to export.')
    if logging == True: 
        print(' > Mesh List:', mesh_list)
        print(' > Armature List:', armature_list)
    
    return mesh_list, armature_list


def get_model_type(uv_bool, normals_bool, rigging_bool, logging):
    if logging == True: print('Finding Model Type...')
    type = None
    if uv_bool == True:
        if normals_bool == True:
            if rigging_bool == True: type = ModelType.xyzuvtbnRigged
            else: type = ModelType.xyzuvtbn
        else:
            if rigging_bool == True: type = ModelType.xyzuvRigged
            else: type = ModelType.xyzuv
    else:
        if normals_bool == True:
            if rigging_bool == True: type = ModelType.xyztbnRigged
            else: type = ModelType.xyztbn
        else:
            if rigging_bool == True: type = ModelType.xyzRigged
            else: type = ModelType.xyz
            
    if logging == True: print(' > Model Type: %s' % type.name)
    return type


def write_to_armature_file(file_path, file_format, bones_dict, logging):
    file_path = file_path + ".armt"
    if logging == True: print('File save location: %s' % file_path)
    
    if file_format == 'Binary':
        if logging == True: print("Writing binary armature data to file...")
        f = open(file_path, 'wb')
        
        f.write(bl_info['version'][0].to_bytes(4, sys.byteorder, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, sys.byteorder, signed=True))
        f.write(len(bones_dict).to_bytes(4, sys.byteorder, signed=True))
        for key in bones_dict:
            bone = bones_dict[key]
            f.write(bone.index.to_bytes(4, sys.byteorder, signed=True))
            f.write(len(bone.name).to_bytes(4, sys.byteorder, signed=True))
            f.write(bone.name.encode('utf-8'))
            matrix_byte_array = numpy.array(bone.local_pose_transform, 'float32')
            matrix_byte_array.tofile(f)
            f.write(len(bone.children_indices).to_bytes(4, sys.byteorder, signed=True))
            indices_byte_array = numpy.array(bone.children_indices, 'int_')
            indices_byte_array.tofile(f)
        
        f.close()
    elif file_format == 'ASCII':
        if logging == True: print("Writing ASCII armature data to file...")
        f = open(file_path, 'w')
        
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('bcnt %d\n' % len(bones_dict))
        for key in bones_dict:
            bone = bones_dict[key]
            f.write('indx %d\n' % bone.index)
            f.write('nsiz %d\n' % len(bone.name))
            f.write('narr %s\n' % bone.name)
            f.write('matx')
            for num in bone.local_pose_transform: f.write(' %f' % num)
            f.write('\n')
            f.write('isiz %d\n' % len(bone.children_indices))
            f.write('iarr')
            for indx in bone.children_indices: f.write(' %d' % indx)
            f.write('\n')
        
        f.close()
    else: raise Error('Invalid file format.')
    

def write_to_model_file(file_path, file_format, model, logging):
    file_path = file_path + ".model"
    if logging == True: print('File save location: %s' % file_path)
    
    if file_format == 'Binary':
        if logging == True: print("Writing binary model data to file...")
        f = open(file_path, 'wb')
        
        f.write(bl_info['version'][0].to_bytes(4, sys.byteorder, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, sys.byteorder, signed=True))
        f.write(model.type.value.to_bytes(4, sys.byteorder, signed=True))
        f.write(model.vertex_stride.to_bytes(4, sys.byteorder, signed=True))
        f.write(model.vertex_count.to_bytes(4, sys.byteorder, signed=True))
        vertices_byte_array = numpy.array(model.vertices, 'float32')
        vertices_byte_array.tofile(f)
        f.write(model.material_count.to_bytes(4, sys.byteorder, signed=True))
        for i in range(model.material_count):
            f.write(len(model.indices_list[i]).to_bytes(4, sys.byteorder, signed=True))
            indices_byte_array = numpy.array(model.indices_list[i], 'int_')
            indices_byte_array.tofile(f)
           
        f.close()
    elif file_format == 'ASCII':
        if logging == True: print("Writing ASCII model data to file...")
        f = open(file_path, 'w')
            
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('mtyp %d\n' % model.type.value)
        f.write('vstr %d\n' % model.vertex_stride)
        f.write('vcnt %d\n' % model.vertex_count)
        f.write('varr')
        for vert in model.vertices: f.write(' %f' % vert)
        f.write('\n')
        f.write('matc %d\n' % model.material_count)
        for i in range(model.material_count):
            f.write('isiz %d\n' % len(model.indices_list[i]))
            f.write('iarr')
            for indx in model.indices_list[i]: f.write(' %d' % indx)
            f.write('\n')
            
        f.close()      
    else: raise Error('Invalid file format.')


def export_model(context, file_path, file_ext, file_format, selection, uv_bool, normals_bool, rigging_bool, logging):
    print('\EXPORTER STARTED')
    print('Exporter Version: %d.%d' % (bl_info['version'][0], bl_info['version'][1]))
    
    try:
        # Fetch all the objects to export
        object_list, armature_list = make_objects_list(selection, logging)
        model_type = get_model_type(uv_bool, normals_bool, rigging_bool, logging)
    
        if rigging_bool == True:
            armature_dict = {}
            # Process all Armature data
            for armature in armature_list:
                armature_dict[armature.name] = process_armature_data(armature.data, logging)
            # Write all Armature data to file
            for key in armature_dict:
                write_to_armature_file(file_path, file_format, armature_dict[key], logging)
            print('ARMATURE EXPORT FINISHED')
        
        for object in object_list:
            # Fetch Object's Armature data (aka. bones_dict) for current object
            if rigging_bool == True:
                bones_dict = None
                for modifier in object.modifiers:
                    if modifier.type == 'ARMATURE':
                        bones_dict = armature_dict[modifier.object.name]
                    break
                if bones_dict == None: raise Error('Object "%s" missing armature.' % object.name)
            
            # Fetch and process Object's Model data
            model = process_model_data(object, model_type, bones_dict, logging)
            if logging == True: model.log()
            
            # Modify file_path if necessary
            filepath = file_path if len(object_list) == 1 else file_path + '_' + object.name.replace(':', '_')
            
            # Write Model data to file
            write_to_model_file(filepath, file_format, model, logging)
            if logging == True: print('Finished exporting %s...' % object.name)
            
        print('MODEL EXPORT FINISHED')
    except Error as e: 
        print('Error:', e)
        
    print('EXPORTER FINISHED')
    return {'FINISHED'}


# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator, Panel


class ExportModel(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_model.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Model"

    # ExportHelper mixin class uses this
    filename_ext = ".model"

    filter_glob: StringProperty(
        default="*.model",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    selection: EnumProperty(
        name="Selection",
        description="Select which objects to export",
        items=(
            ('selected_only',     "Selected Only",       "Export currently selected object(s) only"),
            ('selected_children', "Selected + Children", "Export currently selected object(s) and its childrens"),
            ('all_objects',       "All Objects",         "Export all objects"),
        ),
        default='selected_children',
    )
    
    format: EnumProperty(
        name="Format",
        description="Choose file format",
        items=(
            ('ASCII',  "ASCII",  "Text file format"),
            ('Binary', "Binary", "Binary file format"),
        ),
        default='Binary',
    )
    
    log_bool: BoolProperty(
        name="Log to console",
        description="Logs all data and calls to console",
        default=True,
    )
    
    uv_bool: BoolProperty(
        name="UVs",
        description="Writes UV data in vertex array",
        default=False,
    )
    
    normals_bool: BoolProperty(
        name="Normals",
        description="Writes normal data in vertex array",
        default=False,
    )
    
    rigging_bool: BoolProperty(
        name="Rigging",
        description="Writes bone data in vertex array, and exports armature in .armt file",
        default=True,
    )   
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        layout.label(text='Export Settings:')
        layout.prop(self, 'selection')
        layout.prop(self, 'format')
        layout.prop(self, 'log_bool')
        
        layout.label(text='Include:')
        layout.prop(self, 'uv_bool')
        layout.prop(self, 'normals_bool')
        layout.prop(self, 'rigging_bool')  

    def execute(self, context):
        return export_model(context, self.filepath.replace(self.filename_ext, ''), self.filename_ext, self.format, self.selection, self.uv_bool, self.normals_bool, self.rigging_bool, self.log_bool)
    
    
# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(ExportModel.bl_idname, text="Model Exporter (.model)")


def register():
    bpy.utils.register_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.export_model.some_data('INVOKE_DEFAULT')
    
