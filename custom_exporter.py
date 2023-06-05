import bpy
import sys
import bmesh
import numpy
import mathutils
from bpy_extras.io_utils import axis_conversion

bl_info = {
    "name": "Custom Exporter Add-on",
    "description": "Exports blender data in a more optimized format for graphics APIs such as OpenGL, Vulkan, and DirectX.",
    "author": "Pablo Rascazzi",
    "version": (0, 1, 3),
    "blender": (3, 5, 1),
    "location": "File > Export > Custom Exporter",
    "category": "Import-Export"
}

identifier = 1129529682

#######################################################################################################################
#                                                 Class Definitions                                                   #
#######################################################################################################################
        
class Error(Exception):
    pass


class MeshFlag():
    # 16-Bit Flags
    HAS_POSITION          = 0b0000000000000001
    HAS_COLOR             = 0b0000000000000010
    HAS_UV                = 0b0000000000000100
    HAS_NORMAL            = 0b0000000000001000
    HAS_TANGENT           = 0b0000000000010000
    HAS_BITANGENT         = 0b0000000000100000
    HAS_BONE_WEIGHT       = 0b0000000001000000
    HAS_BONE_WEIGHT_INDEX = 0b0000000010000000


class MeshData():
    def __init__(self):
        self.name = None # Unique mesh ID
        self.flags = None # Bit-flags for what is in the vertex (ie. Colors, UVs, Normals, etc.)
        self.vertex_count = None # Amount of vertices on the mesh
        self.vertex_stride = None # Amount of elements in a single vertex
        
        # Array of all vertices
        self.vertices = []
        
        # Arrays of all vertices separated by data
        self.positions = []
        self.colors = []
        self.uvs = []
        self.normals = []
        self.tangents = []
        self.bitangents = []
        self.bone_weights = []
        self.bone_weights_indices = []
        
        # Array of all submeshes
        self.submeshes = []
    
    def log(self):
        print('MeshData [%s]' % self.name)
        print(' > Flags: %d' % self.flags)
        print(' > Vertex Count: %d' % self.vertex_count)
        print(' > Vertex Stride: %d' % self.vertex_stride)
        print(' > Vertices Array:', self.vertices)
        print(' > Positions Array:', self.positions)
        print(' > Colors Array:', self.colors)
        print(' > UVs Array:', self.uvs)
        print(' > Normals Array:', self.normals)
        print(' > Tangents Array:', self.tangents)
        print(' > Bitangents Array:', self.bitangents)
        print(' > Weights Array:', self.bone_weights)
        print(' > Weight Indices Array:', self.bone_weights_indices)
        print(' > Submesh Count: %d' % len(self.submeshes))
        for i in range(len(self.submeshes)): self.submeshes[i].log(str(i));

   
class SubmeshData():
    def __init__(self):
        self.indices = []
        
    def log(self, name):
        print(' > SubmeshData [%s]' % name)
        print(' == > Index Count: %d' % len(self.indices))
        print(' == > Indices Array:', self.indices)
        
    
#######################################################################################################################
#                                                     Utilities                                                       #
#######################################################################################################################

def f_round(num, precision):
    rounded_num = round(num, precision)
    return 0 if rounded_num == 0 or rounded_num == -0 else rounded_num


def vec_round(vec, precision):
    return list(map(lambda num: f_round(num, precision), vec))


def matrix_to_float16(matrix, precision):
    float16 = []
    float16.extend([f_round(matrix[0][0], precision), f_round(matrix[0][1], precision), f_round(matrix[0][2], precision), f_round(matrix[0][3], precision)])
    float16.extend([f_round(matrix[1][0], precision), f_round(matrix[1][1], precision), f_round(matrix[1][2], precision), f_round(matrix[1][3], precision)])
    float16.extend([f_round(matrix[2][0], precision), f_round(matrix[2][1], precision), f_round(matrix[2][2], precision), f_round(matrix[2][3], precision)])
    float16.extend([f_round(matrix[3][0], precision), f_round(matrix[3][1], precision), f_round(matrix[3][2], precision), f_round(matrix[3][3], precision)])
    return float16


def vec_convert_axis(vector, matrix):
    vector = mathutils.Vector(vector)
    vector.rotate(matrix)
    return vector


#######################################################################################################################
#                                                   Mesh Processing                                                   #
#######################################################################################################################

def process_mesh_data(context, mesh):
    """
    Returns an instance of MeshData() initialized with the processed data fetched from the mesh.
    
    Parameter context: an instance of ExporterProperties().
    Prerequisite: is initialized by Blender when the add-on is executed.
    
    Parameter mesh: an instance of bpy.types.Mesh().
    """
    output_data = MeshData()
    output_data.name = mesh.name
    output_data.flags = process_mesh_flags(context)
    output_data.vertex_stride = process_mesh_vertex_stride(output_data.flags)
    
    # Fetch necessary attributes
    uv_layer = None
    color_layer = None
    if output_data.flags & (MeshFlag.HAS_COLOR) != 0:
        if mesh.vertex_colors.active == None: 
            raise Error("Mesh [%s] is missing an active vertex color attribute." % mesh.name)
        color_layer = mesh.vertex_colors.active.data
    if output_data.flags & (MeshFlag.HAS_UV) != 0:
        if mesh.uv_layers.active == None: 
            raise Error("Mesh [%s] is missing an active uv map attribute." % mesh.name)
        uv_layer = mesh.uv_layers.active.data
    if output_data.flags & (MeshFlag.HAS_NORMAL) != 0:
        mesh.calc_normals_split()
    if output_data.flags & (MeshFlag.HAS_TANGENT) != 0 or output_data.flags & (MeshFlag.HAS_BITANGENT) != 0:
        mesh.calc_tangents()
    
    vert_dict = {}
    next_index = 0
    previous_face_length = 0
    for face in mesh.polygons:
        face_indices = []
        
        # Processing face vertices...
        for vert_id, loop_id in zip(face.vertices, face.loop_indices):
            loop = mesh.loops[loop_id]
            vert = []
            # Process vertex position
            position = mesh.vertices[vert_id].co
            position = vec_round(vec_convert_axis(position, context.axis_conversion_matrix) if context.apply_axis_conversion else position, context.precision)
            vert.extend(position)
            # Process vertex color
            if context.mesh_incl_colors == True:
                color = vec_round(color_layer[loop_id].color[:3], context.precision)
                vert.extend(color)
            # Process vertex UV
            if context.mesh_incl_uvs == True:
                uv = vec_round(uv_layer[loop_id].uv[:2], context.precision)
                vert.extend(uv)
            # Process vertex normal
            if context.mesh_incl_normals == True:
                normal = loop.normal
                normal = vec_round(vec_convert_axis(normal, context.axis_conversion_matrix) if context.apply_axis_conversion else normal, context.precision)
                vert.extend(normal)
            # Process vertex tangent
            if context.mesh_incl_tangents == True:
                tangent = loop.tangent
                tangent = vec_round(vec_convert_axis(tangent, context.axis_conversion_matrix) if context.apply_axis_conversion else tangent, context.precision)
                vert.extend(tangent)
            # Process vertex bitangent
            if context.mesh_incl_bitangents == True:
                bitangent = loop.bitangent
                bitangent = vec_round(vec_convert_axis(bitangent, context.axis_conversion_matrix) if context.apply_axis_conversion else bitangent, context.precision)
                vert.extend(bitangent)
            # Process vertex bone weights and bone weights indices
            if context.mesh_incl_rigging == True:
                raise Error("Rigging information for meshes is not yet supported.")
                pass # TODO
            
            # Check if current vertex does not exists in vertex dictionary
            vert_key = str(vert)
            if vert_key not in vert_dict:
                vert_dict[vert_key] = next_index
                
                # Add new vertex to correct array(s)
                if context.mesh_buffer_format == 'interleaved':
                    # Append vertex into vertices array
                    output_data.vertices.extend(vert)
                elif context.mesh_buffer_format == 'separate':
                    # Append vertex into correct attribute array
                    next_vert_index = 0
                    if output_data.flags & (MeshFlag.HAS_POSITION) != 0:
                        output_data.positions.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_COLOR) != 0:
                        output_data.colors.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_UV) != 0:
                        output_data.uvs.extend(vert[next_vert_index:next_vert_index+2])
                        next_vert_index += 2
                    if output_data.flags & (MeshFlag.HAS_NORMAL) != 0:
                        output_data.normals.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_TANGENT) != 0:
                        output_data.tangents.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_BITANGENT) != 0:
                        output_data.bitangents.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_BONE_WEIGHT) != 0:
                        output_data.bone_weights.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if output_data.flags & (MeshFlag.HAS_BONE_WEIGHT_INDEX) != 0:
                        output_data.bone_weights_indices.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                else: raise Error('Invalid vertex buffer format.')
                
                # Append new vertex's index to face indices list
                face_indices.append(next_index)
                next_index += 1
            else:
                # Append existing vertex's index to face indices list
                face_indices.append(vert_dict[vert_key]) 
            
        #Processing face indices...    
        if len(face_indices) < 3:
            print('Warning: Points and lines are not supported.')
        else:
            primitive_list = []
            # Pad submesh array if smaller than current material index
            for n in range(len(output_data.submeshes), face.material_index+1):  
                output_data.submeshes.append(SubmeshData())
            
            # Process indices as triangle lists
            if context.mesh_primitive_topology == 'triangle_list':
                num_triangles = len(face_indices)-2
                list_1 = [0] * num_triangles       # [0, 0, 0, 0, ...]
                list_2 = range(1, 1+num_triangles) # [1, 2, 3, 4, ...]
                list_3 = range(2, 2+num_triangles) # [2, 3, 4, 5, ...]
                primitive_list = [face_indices[i] for i in list(sum(zip(list_1, list_2, list_3), ()))]
            
            # Process indices as triangle strips
            elif context.mesh_primitive_topology == 'triangle_strip':
                # Concatenate new triangle strip with previous strips
                if len(output_data.submeshes[face.material_index].indices) > 0:
                    # If previous face length is odd, then make it even before concatenating
                    if (previous_face_length % 2) == 1: 
                        primitive_list.append(output_data.submeshes[face.material_index].indices[-1])
                    # Add degenerate triangle to concatenate
                    primitive_list.extend([output_data.submeshes[face.material_index].indices[-1], face_indices[0]])
                # Add face indices to primitive list
                if len(face_indices) == 3:
                    primitive_list.extend([face_indices[0], face_indices[1], face_indices[2]])
                    previous_face_length = 3
                if len(face_indices) == 4:
                    primitive_list.extend([face_indices[0], face_indices[1], face_indices[3], face_indices[2]])
                    previous_face_length = 4
                if len(face_indices) > 4:
                    for i in range(1, len(face_indices)): 
                        primitive_list.extend([face_indices[0], face_indices[i]])
                    previous_face_length = len(range(1, len(face_indices)) *2)
            
            # Process indices as triangle strips separated by a primitive restart index
            elif context.mesh_primitive_topology == 'triangle_strip_restart':
                # Add primitive restart index
                if len(output_data.submeshes[face.material_index].indices) > 0:
                    primitive_list.append(context.mesh_primitive_restart_index)
                # Add face indices to primitive list
                if len(face_indices) == 3:
                    primitive_list.extend([face_indices[0], face_indices[1], face_indices[2]])
                if len(face_indices) == 4:
                    primitive_list.extend([face_indices[0], face_indices[1], face_indices[3], face_indices[2]])
                if len(face_indices) > 4:
                    for i in range(1, len(face_indices)): 
                        primitive_list.extend([face_indices[0], face_indices[i]])
            else: raise Error('Invalid mesh primitive topology.')
            
            # Append current face's indices to current submesh
            output_data.submeshes[face.material_index].indices.extend(primitive_list)
            
    # Set output vertex count as the length of vertex dictionary
    output_data.vertex_count = len(vert_dict)
    if context.logging == True: output_data.log()
    
    # Return output mesh data
    return output_data


def process_mesh_flags(context):
    flags = MeshFlag.HAS_POSITION
    if context.mesh_incl_colors:     flags |= MeshFlag.HAS_COLOR
    if context.mesh_incl_uvs:        flags |= MeshFlag.HAS_UV
    if context.mesh_incl_normals:    flags |= MeshFlag.HAS_NORMAL
    if context.mesh_incl_tangents:   flags |= MeshFlag.HAS_TANGENT
    if context.mesh_incl_bitangents: flags |= MeshFlag.HAS_BITANGENT
    if context.mesh_incl_rigging:    flags |= MeshFlag.HAS_BONE_WEIGHT
    if context.mesh_incl_rigging:    flags |= MeshFlag.HAS_BONE_WEIGHT_INDEX
    return flags


def process_mesh_vertex_stride(flags):
    vertex_stride = 0
    # Calculate vertex stride using flags
    if flags & (MeshFlag.HAS_POSITION) != 0:          vertex_stride += 3
    if flags & (MeshFlag.HAS_COLOR) != 0:             vertex_stride += 3
    if flags & (MeshFlag.HAS_UV) != 0:                vertex_stride += 2
    if flags & (MeshFlag.HAS_NORMAL) != 0:            vertex_stride += 3
    if flags & (MeshFlag.HAS_TANGENT) != 0:           vertex_stride += 3
    if flags & (MeshFlag.HAS_BITANGENT) != 0:         vertex_stride += 3
    if flags & (MeshFlag.HAS_BONE_WEIGHT) != 0:       vertex_stride += 3
    if flags & (MeshFlag.HAS_BONE_WEIGHT_INDEX) != 0: vertex_stride += 3
    # Return calculated vertex stride
    return vertex_stride
    
    
#######################################################################################################################
#                                             Object Selection Processing                                             #
#######################################################################################################################

def process_export_selection(context):
    """
    Returns a list of bpy.types.Object() of type 'MESH' based on the export context's object selection.
    
    Parameter context: an instance of ExporterProperties().
    Prerequisite: is initialized by Blender when the add-on is executed.
    """
    selected_objects = []
    # Recursively find all child objects
    def recursive_search(object):
        for child in object.children:
            if child == None: continue
            recursive_search(child)
        # Check for duplicates before adding to the list
        if object not in selected_objects and object.type == 'MESH':
            selected_objects.append(object)
    
    # Find selected objects
    if context.object_selection == 'selected_only':
        selected_objects = [object for object in bpy.context.selected_objects if object.type == 'MESH']
    elif context.object_selection == 'selected_children':
        selected_objects = [object for object in bpy.context.selected_objects if object.type == 'MESH']
        # Recursively search for children
        for object in bpy.context.selected_objects:
            if object == None:  continue 
            recursive_search(object)
    elif context.object_selection == 'all_objects':
        selected_objects = [object for object in bpy.context.visible_objects if object.type == 'MESH']
    else: raise Error('Unsupported object selection.')
    
    # Check for if nothing is to be exported
    if context.incl_mesh == False and context.incl_armt == False and context.incl_anim == False:
        raise Error("Nothing to export. Select properties to export in the 'Export Includes'.")
    if not selected_objects: 
        raise Error('Nothing to export. Select object to export from the Scene.')
    
    # Return all selected objects
    return selected_objects

#######################################################################################################################
#                                                     File Writing                                                    #
#######################################################################################################################

def write_to_mesh_file(context, mesh):
    """
    Write all mesh data to a binary file format.
    
    Parameter context: an instance of ExporterProperties().
    Prerequisite: is initialized by Blender when the add-on is executed.
    
    Parameter mesh: an instance of MeshData().
    Prerequisite: is initialized with valid data depending on context.
    """
    # Fetch filepath
    filepath = context.filepath + ("" if context.filepath.endswith('\\') else "_") + mesh.name.replace('.', '_').replace(':', '_') + ".mesh"
    
    # Open file for writing
    f = open(filepath, 'wb')
    
    # File Header
    f.write(identifier.to_bytes(4, context.byte_order, signed=True))
    f.write(bl_info['version'][0].to_bytes(4, context.byte_order, signed=True))
    f.write(bl_info['version'][1].to_bytes(4, context.byte_order, signed=True))
    
    # Mesh Header
    f.write(mesh.flags.to_bytes(4, context.byte_order, signed=False))
    if context.mesh_buffer_format == 'interleaved':
        f.write((0).to_bytes(4, context.byte_order, signed=False))
    elif context.mesh_buffer_format == 'separate':
        f.write((1).to_bytes(4, context.byte_order, signed=False))
    else: raise Error('Unsupported vertex buffer format.')
    if context.mesh_primitive_topology == 'triangle_list':
        f.write((0).to_bytes(4, context.byte_order, signed=False))
    elif context.mesh_primitive_topology == 'triangle_strip':
        f.write((1).to_bytes(4, context.byte_order, signed=False))
    elif context.mesh_primitive_topology == 'triangle_strip_restart':
        f.write((2).to_bytes(4, context.byte_order, signed=False))
    else: raise Error('Unsupported mesh primitive topology.')
    f.write(mesh.vertex_stride.to_bytes(4, context.byte_order, signed=False))
    f.write(mesh.vertex_count.to_bytes(4, context.byte_order, signed=False))
    f.write(len(mesh.submeshes).to_bytes(4, context.byte_order, signed=False))
    
    # Vertex Data...
    if context.mesh_buffer_format == 'interleaved':
        # Write interleaved vertex buffer to file
        vertices_bytes = numpy.array(mesh.vertices, 'float32')
        if context.swap_byte_order == True: vertices_bytes.byteswap()
        vertices_bytes.tofile(f)
    elif context.mesh_buffer_format == 'separate':
        # Write separate vertex buffers to file
        if mesh.flags & (MeshFlag.HAS_POSITION) != 0:
            positions_bytes = numpy.array(mesh.positions, 'float32')
            if context.swap_byte_order == True: positions_bytes.byteswap()
            positions_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_COLOR) != 0:
            colors_bytes = numpy.array(mesh.colors, 'float32')
            if context.swap_byte_order == True: colors_bytes.byteswap()
            colors_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_UV) != 0:
            uvs_bytes = numpy.array(mesh.uvs, 'float32')
            if context.swap_byte_order == True: uvs_bytes.byteswap()
            uvs_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_NORMAL) != 0:
            normals_bytes = numpy.array(mesh.normals, 'float32')
            if context.swap_byte_order == True: normals_bytes.byteswap()
            normals_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_TANGENT) != 0:
            tangents_bytes = numpy.array(mesh.tangents, 'float32')
            if context.swap_byte_order == True: tangents_bytes.byteswap()
            tangents_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_BITANGENT) != 0:
            bitangents_bytes = numpy.array(mesh.bitangents, 'float32')
            if context.swap_byte_order == True: bitangents_bytes.byteswap()
            bitangents_bytes.tofile(f)
        if mesh.flags & (MeshFlag.HAS_BONE_WEIGHT_INDEX) != 0:
            weight_indices_byte_array  = numpy.array(mesh.bone_weights_indices, 'uint32')
            if context.swap_byte_order == True: weight_indices_byte_array .byteswap()
            weight_indices_byte_array .tofile(f)
        if mesh.flags & (MeshFlag.HAS_BONE_WEIGHT) != 0:
            weights_byte_array = numpy.array(mesh.bone_weights, 'float32')
            if swap_bytes == True: weights_byte_array.byteswap()
            weights_byte_array.tofile(f)
    else: raise Error('Unsupported vertex buffer format.')
    
    # Submesh Data
    for submesh in mesh.submeshes:
        f.write(len(submesh.indices).to_bytes(4, context.byte_order, signed=False))
        indices_bytes = numpy.array(submesh.indices, 'uint32')
        if context.swap_byte_order == True: indices_bytes.byteswap()
        indices_bytes.tofile(f)
     
    # Close file
    f.close()
    
    # Print file saved location as success confirmation
    print('Mesh [%s] file saved location: %s' % (mesh.name, filepath))
    

#######################################################################################################################
#                                                  Exporter Execution                                                 #
#######################################################################################################################

def execute_exporter(context):
    print('\nEXPORT STARTED')
    
    try:
        # Compare system byte-order with exporter byte-order
        if (sys.byteorder == 'little' and context.byte_order != 'little') or (sys.byteorder == 'big' and context.byte_order != 'big'): context.swap_byte_order = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Unsupported system byte order.')
        
        # Create axis conversion matrix
        context.axis_conversion_matrix = axis_conversion(from_forward='-Y', from_up='Z', to_forward=context.axis_conversion_forward, to_up=context.axis_conversion_up).to_4x4()
        
        # Set mode to Object
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Fetch all selected objects
        all_selected_objects = process_export_selection(context)
        
        # Process mesh data from all selected objects
        for object in all_selected_objects:
            try:
                mesh_data = process_mesh_data(context, object.data)
                write_to_mesh_file(context, mesh_data)
            except Error as e:
                print('Error:', e)
        
    except Error as e:
        print('Error:', e)

    print('EXPORT FINISHED')
    return {'FINISHED'}


#######################################################################################################################
#                                                Blender Initialization                                               #
#######################################################################################################################

# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, IntProperty, EnumProperty
from bpy.types import Operator

class ExporterProperties(bpy.types.PropertyGroup):
    # List of exporter properties
    # Main properties
    object_selection: EnumProperty(
        name="Selection",
        description="Select which objects to export",
        items=(
            ('selected_only',     "Selected Only",       "Export currently selected object(s) only"),
            ('selected_children', "Selected + Children", "Export currently selected object(s) and its childrens"),
            ('all_objects',       "All Objects",         "Export all objects"),
        ),
        default='selected_children',
    )
    incl_mesh: BoolProperty(
        name="Mesh",
        description="Writes mesh data and exports it in .mesh file",
        default=True,
    )
    incl_armt: BoolProperty(
        name="Armature",
        description="Writes bone data in a hierarchy list and exports it in .armt file",
        default=True,
    )
    incl_anim: BoolProperty(
        name="Animation",
        description="Writes animation data and exports it in .anim file",
        default=True,
    )
    apply_axis_conversion: BoolProperty(
        name="Apply",
        description="Applies global axis conversion using selected values",
        default=True,
    )
    axis_conversion_up: EnumProperty(
        name="Up",
        items=(
            ("X",   "X", "", 0),
            ("Y",   "Y", "OpenGL & Vulkan & DirectX", 1),
            ("Z",   "Z", "", 2),
            ("-X", "-X", "", 3),
            ("-Y", "-Y", "", 4),
            ("-Z", "-Z", "", 5),
        ),
        default="Y",
    )
    axis_conversion_forward: EnumProperty(
        name="Forward",
        items=(
            ("X",   "X", "", 0),
            ("Y",   "Y", "", 1),
            ("Z",   "Z", "Vulkan & DirectX", 2),
            ("-X", "-X", "", 3),
            ("-Y", "-Y", "", 4),
            ("-Z", "-Z", "OpenGL", 5),
        ),
        default="Z",
    )
    axis_conversion_matrix = None
    
    # Mesh properties
    mesh_buffer_format: EnumProperty(
        name="Vertex Buffer",
        description="Choose the vertex buffer format",
        items=(
            ('interleaved',  "Interleaved", "Vertex data will be stored into one buffer."),
            ('separate', "Separate", "Vertex data will be stored into separate buffers for each attributes."),
        ),
        default='interleaved',
    )
    mesh_primitive_topology: EnumProperty(
        name="Primitive Topology",
        description="",
        items=(
            ('triangle_list', 'Triangle List', "Index data will be stored as a list of triangles."),
            ('triangle_strip', 'Triangle Strip', "Index data will be stored as a list of triangle strips."),
            ('triangle_strip_restart', 'Triangle Strip (Primitive Restart)', "Index data will be stored as a list of triangle strips with a primite restart index of 0xFFFFFFFF to separate each strips."),
        ),
        default="triangle_list",
    )
    mesh_primitive_restart_index = 0xFFFFFFFF
    mesh_incl_colors: BoolProperty(
        name="Colors",
        description="Writes color data in vertex buffer",
        default=True,
    )
    mesh_incl_uvs: BoolProperty(
        name="UVs",
        description="Writes UV (Aka. Texture Coordinates) data in vertex buffer",
        default=False,
    )
    mesh_incl_normals: BoolProperty(
        name="Normals",
        description="Writes normal data in vertex buffer",
        default=False,
    )
    mesh_incl_tangents: BoolProperty(
        name="Tangents",
        description="Writes tangent data in vertex buffer",
        default=False,
    )
    mesh_incl_bitangents: BoolProperty(
        name="Bitangents",
        description="Writes bitangent data in vertex buffer",
        default=False,
    )
    mesh_incl_rigging: BoolProperty(
        name="Rigging",
        description='Writes bone weight and index in vertex buffer.\nCan only be exported if vertex buffer format is set to "Separate Buffers"',
        default=False,
    )
    
    # Extra properties
    byte_order: EnumProperty(
        name="Byte Order",
        description="Choose byte order for binary file",
        items=(
            ("little",  "Little-Endian",  "Little-Endian byte order"),
            ("big", "Big-Endian", "Big-Endian byte order"),
        ),
        default="little",
    )
    precision: IntProperty(
        name="Precision",
        description="Rounding precision applied to all float values",
        default=6, min=2, max=8
    )
    logging: BoolProperty(
        name="Log to console",
        description="Logs all data, and calls, to console",
        default=True
    )
    
    # Other properties
    tabs: EnumProperty(
        items=(
            ("MAIN",  "main",  "Main Options"      , 0),
            ("GEOM",  "geom",  "Mesh Options"      , 1),
            ("ARMT",  "armt",  "Armature Options"  , 2),
            ("ANIM",  "anim",  "Animation Options" , 3),
            ("EXTRA", "extra", "Additional Options", 4),
        ),
        default="MAIN",
    )
    filepath: StringProperty(
        name="Filepath",
        default="",
    )
    swap_byte_order: BoolProperty(
        default=False
    )
       

class CUSTOM_EXPORTER_OT_main(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "custom_exporter.main"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export"
    filename_ext = "" # ExportHelper mixin class uses this

    # Filters files so that only MESH, ARMT, and ANIM files are shown
    filter_glob: StringProperty(
        default="*.mesh; *.armt; *.anim",
        options={'HIDDEN'}
    )
    
    # Draws the export menu
    def draw(self, context):
        # Draws the tab buttons
        row = self.layout.row()
        row.prop(context.scene.custom_exporter, 'tabs', expand=True)
        
        # Changes the decoration of the layout
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        
        # Draws the main tab
        if context.scene.custom_exporter.tabs == 'MAIN':
            main_box = self.layout.box()
            main_box.label(text="Main Options",icon='WORLD')
            main_box.prop(context.scene.custom_exporter, 'object_selection')
            
            include_box = self.layout.box()
            include_box.label(text="Export Includes",icon='EXPORT')
            include_box.prop(context.scene.custom_exporter, 'incl_mesh')
            include_box.prop(context.scene.custom_exporter, 'incl_armt')
            include_box.prop(context.scene.custom_exporter, 'incl_anim')
            
            conversion_box = self.layout.box()
            conversion_box.label(text="Global Axis Conversion", icon="OBJECT_ORIGIN")
            conversion_box.prop(context.scene.custom_exporter, 'axis_conversion_up')
            conversion_box.prop(context.scene.custom_exporter, 'axis_conversion_forward')
            conversion_box.prop(context.scene.custom_exporter, 'apply_axis_conversion')
          
        # Draws the geometry tab
        elif context.scene.custom_exporter.tabs == 'GEOM':
            geometry_box = self.layout.box()
            geometry_box.label(text="Mesh Options",icon='MESH_DATA')
            geometry_box.prop(context.scene.custom_exporter, 'mesh_buffer_format')
            geometry_box.prop(context.scene.custom_exporter, 'mesh_primitive_topology')
            include_box = self.layout.box()
            include_box.label(text="Vertex Includes",icon='EXPORT')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_colors')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_uvs')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_normals')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_tangents')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_bitangents')
            include_box.prop(context.scene.custom_exporter, 'mesh_incl_rigging')
           
        # Draws the armature tab 
        elif context.scene.custom_exporter.tabs == 'ARMT':
            armature_box = self.layout.box()
            armature_box.label(text="Armature Options",icon='ARMATURE_DATA')
          
        # Draws the animation tab  
        elif context.scene.custom_exporter.tabs == 'ANIM':
            animation_box = self.layout.box()
            animation_box.label(text="Animation Options",icon='ACTION')
          
        # Draws the extra tab  
        elif context.scene.custom_exporter.tabs == 'EXTRA':
            extra_box = self.layout.box()
            extra_box.label(text="Additional Options",icon='MODIFIER')
            extra_box.prop(context.scene.custom_exporter, 'byte_order')
            extra_box.prop(context.scene.custom_exporter, 'precision')
            extra_box.prop(context.scene.custom_exporter, 'logging')

    # Executes the export
    def execute(self, context):
        context.scene.custom_exporter.filepath = self.filepath.replace(self.filename_ext, '')
        return execute_exporter(context.scene.custom_exporter)


# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(CUSTOM_EXPORTER_OT_main.bl_idname, text="Custom Exporter (.mesh, .amrt, .anim)")

# Register and add to the "file selector" menu (required to use F3 search "Text Export Operator" for quick access)
def register():
    # Register the ExporterProperties class
    bpy.utils.register_class(ExporterProperties)
    bpy.types.Scene.custom_exporter = bpy.props.PointerProperty(type=ExporterProperties)
    
    # Register the Main operator
    bpy.utils.register_class(CUSTOM_EXPORTER_OT_main)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    # Unregister the ExporterProperties class
    bpy.utils.unregister_class(ExporterProperties)
    del bpy.types.Scene.custom_exporter
    
    # Unregister the Main operator
    bpy.utils.unregister_class(CUSTOM_EXPORTER_OT_main)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()

    # Test call
    bpy.ops.custom_exporter.main('INVOKE_DEFAULT')