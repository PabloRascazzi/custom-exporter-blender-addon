import bpy
import mathutils
import math
import sys
import numpy
from enum import Enum
from bpy_extras.io_utils import axis_conversion

bl_info = {
    "name": "OpenGL Exporter Add-on",
    "description": "Exports blender data in a more optimized format for OpenGL.",
    "author": "Pablo Rascazzi",
    "version": (0, 7, 4),
    "blender": (2, 92, 0),
    "location": "File > Export > OpenGL Exporter",
    "category": "Import-Export"
}

identifier = 1129529682

conversion = axis_conversion(from_forward='Y', from_up='Z', to_forward='-Z', to_up='Y').to_4x4()

#######################################################################################################################
#                                                 Class Definitions                                                   #
#######################################################################################################################

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


class ExportContext():
    def __init__(self):
        # General Settings
        self.object_selection = None
        self.file_format = None
        self.byte_order = None
        self.precision = None
        self.flip_axis = None
        self.logging = None
        # Export Includes
        self.include_model = None
        self.include_armt = None
        self.include_anim = None
        # Model Settings
        self.model_buffer_format = None
        self.include_uvs = None
        self.include_normals = None
        self.include_bones = None
        # Armature Settings
        self.armt_matrix = None
        # Animation Settings
        self.anim_time_format = None
        self.anim_export_frames = None
        self.anim_frame_interval = None
        
    def log(self):
        print('Exporter Context:')
        print(' > Exporter Version: %d.%d.%d' % (bl_info['version'][0], bl_info['version'][1], bl_info['version'][2]))
        print(' > Export File Format: %s' % (self.file_format))
        print(' > Export Byte Order: %s' % (self.byte_order))
        print(' > Export Precision: %s' % (self.precision))
        print(' > Object Selection: %s' % (self.object_selection))
        print(' > Convert to OpenGL Axis: %s' % (self.flip_axis))
        print(' > Logging: %s' % (self.logging))
        print(' > Export Includes:')
        print(' == > Include Models: %s' % (self.include_model))
        print(' == > Include Armatures: %s' % (self.include_armt))
        print(' == > Include Animations: %s' % (self.include_anim))
        print(' > Model Settings:')
        print(' == > Buffer Format: %s' % (self.model_buffer_format))
        print(' == > Include UV: %s' % (self.include_uvs))
        print(' == > Include Normals: %s' % (self.include_normals))
        print(' == > Include Bones: %s' % (self.include_bones))
        print(' > Armature Settings:')
        print(' == > Matrix Type: %s' % (self.armt_matrix))
        print(' > Animation Settings:')
        print(' == > Time Format: %s' % (self.anim_time_format))
        print(' == > Export Frames: %s' % (self.anim_export_frames))
        print(' == > Frame Interval: %s' % (self.anim_frame_interval))
        

class Model():  
    def __init__(self, type, vertex_stride, vertex_count, vertices, material_count, indices_list, uvs = None, normals = None, weight_indices = None, weights = None):
        self.type = type
        self.vertex_stride = vertex_stride
        self.vertex_count = vertex_count
        self.vertices = vertices # vertices will store all the data if buffer_format is set to single_buffer
        self.uvs = uvs
        self.normals = normals
        self.weight_indices = weight_indices
        self.weights = weights
        self.material_count = material_count
        self.indices_list = indices_list # A list of indices array
    
    def log(self, buffer_format):
        print(' > Model Type: %s' % self.type.name)
        print(' == > Vertex Stride: %d' % self.vertex_stride)
        print(' == > Vertex Count: %d' % self.vertex_count)
        print(' == == > Vertices Array:', self.vertices)
        if buffer_format == 'separate_buffers':
            print(' == == > UVs Array:', self.uvs)
            print(' == == > Normals Array:', self.normals)
            print(' == == > Weight Indices Array:', self.weight_indices)
            print(' == == > Weights Array:', self.weights)
        print(' == > Material Count: %d' % self.material_count)
        for i in range(self.material_count):
            print(' == == > Indices[%d] Size: %d' % (i, len(self.indices_list[i])))
            print(' == == > Indices[%d] Array:' % i, self.indices_list[i])
        

class BoneInfo():
    def __init__(self):
        self.index = None
        self.name = None
        self.local_bind_transform = None
        self.local_bind_transform_transposed = None
        self.local_bind_transform_transposed_converted = None
        self.inverse_bind_transform = None
        self.inverse_bind_transform_converted = None
        self.children = []
        
    def get_children_indices(self):
        children_indices = []
        for child in self.children: 
            children_indices.append(child.index)
        return children_indices
        
    def calc_inverse_bind_transform(self, flip_axis, parent_bind_transform):
        bind_transform = parent_bind_transform @ self.local_bind_transform
        self.inverse_bind_transform = bind_transform.inverted()
        for child in self.children:
            child.calc_inverse_bind_transform(flip_axis, bind_transform)
        
    def log(self, precision):
        print(' > Bone Name: %s' % self.name)
        print(' == > Index: %d' % self.index)
        print(' == > Children Indices:', self.get_children_indices())
        if self.local_bind_transform != None:
            local_bind_transform_arr = matrix_to_float16(self.local_bind_transform, precision)
            print(' == > Local Bind Transform:')
            print('         ', local_bind_transform_arr[0:4])
            print('         ', local_bind_transform_arr[4:8])
            print('         ', local_bind_transform_arr[8:12])
            print('         ', local_bind_transform_arr[12:16])
        if self.inverse_bind_transform != None:
            inverse_bind_transform_arr = matrix_to_float16(self.inverse_bind_transform, precision)
            print(' == > Inverse Bind Transform:')
            print('         ', inverse_bind_transform_arr[0:4])
            print('         ', inverse_bind_transform_arr[4:8])
            print('         ', inverse_bind_transform_arr[8:12])
            print('         ', inverse_bind_transform_arr[12:16])


class AnimationKeyframe():
    def __init__(self):
        self.timestamp = None
        self.bone_curves = []
        
    def log(self, precision):
        print(' > Timestamp: %f' % self.timestamp)
        print(' > Bone Curves:')
        for curve in self.bone_curves: curve.log(precision)


class BoneTransform():
    def __init__(self):
        self.index = None
        self.name = None
        self.matrix = None
        self.location = [0,0,0]
        self.scale = [0,0,0]
        self.quaternion = [0,0,0,0]
        
    def log(self, precision):
        print(' > Bone Name: %s' % self.name)
        print(' == > Bone Index: %d' % self.index)
        if self.matrix != None:
            matrix_arr = matrix_to_float16(self.matrix, precision)
            print(' == > Local Matrix:')
            print('         ', matrix_arr[0:4])
            print('         ', matrix_arr[4:8])
            print('         ', matrix_arr[8:12])
            print('         ', matrix_arr[12:16])
        print(' == > Location:   ', self.location)
        print(' == > Scale:      ', self.scale)
        print(' == > Quaternion: ', self.quaternion)
    

#######################################################################################################################
#                                                     Utilities                                                       #
#######################################################################################################################

def f_round(num, precision):
    rnd_num = round(num, precision)
    return 0 if rnd_num == 0 or rnd_num == -0 else rnd_num
    

def matrix_to_float16(matrix, precision):
    float16 = []
    float16.extend([f_round(matrix[0][0], precision), f_round(matrix[0][1], precision), f_round(matrix[0][2], precision), f_round(matrix[0][3], precision)])
    float16.extend([f_round(matrix[1][0], precision), f_round(matrix[1][1], precision), f_round(matrix[1][2], precision), f_round(matrix[1][3], precision)])
    float16.extend([f_round(matrix[2][0], precision), f_round(matrix[2][1], precision), f_round(matrix[2][2], precision), f_round(matrix[2][3], precision)])
    float16.extend([f_round(matrix[3][0], precision), f_round(matrix[3][1], precision), f_round(matrix[3][2], precision), f_round(matrix[3][3], precision)])
    return float16
    

#######################################################################################################################
#                                                  Data Processing                                                    #
#######################################################################################################################

def process_armature_data(export_context, armature):
    if export_context.logging == True: print("Processing armature data...")   
    bones_dict = {} # dict of bones with bone names as key (bone.name == vertex_group.name)
    
    root_bone = None
    next_index = 0
    
    # Create BoneInfo and calculate local_bind_transform from armature data
    for bone in armature.bones:
        new_bone = BoneInfo()
        new_bone.index = next_index
        new_bone.name = bone.name
        
        if bone.parent == None: 
            new_bone.local_bind_transform = conversion @ bone.matrix_local if export_context.flip_axis == True else bone.matrix_local
            root_bone = new_bone
        else: 
            new_bone.local_bind_transform = bone.parent.matrix_local.inverted() @ bone.matrix_local.copy()
            bones_dict[bone.parent.name].children.append(new_bone)
        
        bones_dict[bone.name] = new_bone
        next_index += 1
        
    # Recursively call calc_inverse_bind_transform() to calculate all inverse_bind_transform
    root_bone.calc_inverse_bind_transform(export_context.flip_axis, mathutils.Matrix().Identity(4))

    # Log BoneInfo dictionary
    if export_context.logging == True:  
        for key in bones_dict: bones_dict[key].log(export_context.precision)
            
    return bones_dict


def process_animation_data(export_context, armature, bones_dict):
    if export_context.logging == True: print("Processing animation data...")
    action = armature.animation_data.action
    fps = bpy.context.scene.render.fps
    
    if export_context.anim_time_format == 'frames': frame_range = [float(action.frame_range[0]), float(action.frame_range[1])]
    elif export_context.anim_time_format == 'seconds': frame_range = [round(action.frame_range[0]/fps, 8), round(action.frame_range[1]/fps, 8)]
    else: raise Error('Invalid animation time format.')
    
    frame_times = []
    anim_frames = []

    if export_context.anim_export_frames == 'action_pose_markers':
        for marker in action.pose_markers: frame_times.append(marker.frame)
    elif export_context.anim_export_frames == 'keyframe_points':
        for fcurve in action.fcurves: frame_times.extend([int(time.co[0]) for time in fcurve.keyframe_points if int(time.co[0]) not in frame_times]) 
    elif export_context.anim_export_frames == 'interval':
        frame_times = list(range(int(action.frame_range[0]), int(action.frame_range[1]+1), export_context.anim_frame_interval))
    else: raise Error('Invalid animation export frames.')
    frame_times.sort()
    
    if export_context.logging == True: 
        print('Frame Range:', frame_range)
        print('Frame Times:', frame_times)
    
    for frame in frame_times:
        bpy.context.scene.frame_set(frame)
        
        new_frame = AnimationKeyframe()
        if export_context.anim_time_format == 'frames': new_frame.timestamp = frame
        elif export_context.anim_time_format == 'seconds': new_frame.timestamp = (round(frame/fps, 8))
        else: raise Error('Invalid animation time format.')
        
        for pose_bone in armature.pose.bones:
            if pose_bone.name not in bones_dict:
                print('Warning: PoseBone "%s" missing from armature "%s".' %(pose_bone.name, armature.name))
                continue
            
            new_curve = BoneTransform()
            new_curve.name = pose_bone.name
            new_curve.index = bones_dict[pose_bone.name].index
            
            if pose_bone.parent == None:
                pose_bone_matrix = conversion @ pose_bone.matrix if export_context.flip_axis else pose_bone.matrix
            else:
                pose_bone_matrix = pose_bone.parent.matrix.inverted() @ pose_bone.matrix
            pose_bone_location, pose_bone_quaternion, pose_bone_scale = pose_bone_matrix.decompose()
            
            new_curve.matrix = pose_bone_matrix
            new_curve.location = [f_round(pose_bone_location.x, export_context.precision), f_round(pose_bone_location.y, export_context.precision), f_round(pose_bone_location.z, export_context.precision)]
            new_curve.scale = [f_round(pose_bone_scale.x, export_context.precision), f_round(pose_bone_scale.y, export_context.precision), f_round(pose_bone_scale.z, export_context.precision)]
            new_curve.quaternion = [f_round(pose_bone_quaternion.w, export_context.precision), f_round(pose_bone_quaternion.x, export_context.precision), f_round(pose_bone_quaternion.y, export_context.precision), f_round(pose_bone_quaternion.z, export_context.precision)]
            
            new_frame.bone_curves.append(new_curve)
        anim_frames.append(new_frame)
            
    if export_context.logging == True: 
        print('Animation Frames:')
        for anim_frame in anim_frames: anim_frame.log(export_context.precision)
        
    return frame_range, anim_frames


def process_model_data(export_context, object, model_type, bones_dict):
    if export_context.logging == True: print("Processing model data...")
    
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
    uvs = []
    normals = []
    weights = []
    weight_indices = []
    indices = [[]]
    
    next_index = 0
    for face in mesh.polygons:
        tmp_indices = []

        # processing vertices per faces...
        for li in face.loop_indices:
            vi = loop[li].vertex_index # vertex Index
            vert = []
            position = mathutils.Vector([mesh.vertices[vi].co.x, mesh.vertices[vi].co.y, mesh.vertices[vi].co.z])
            if export_context.flip_axis == True: position.rotate(conversion)
            vert.extend([round(position.x, export_context.precision), round(position.y, export_context.precision), round(position.z, export_context.precision)])
            if 'uv' in model_type.name:
                vert.append(round(uv_layer[li].uv.x, export_context.precision))
                vert.append(round(uv_layer[li].uv.y, export_context.precision))
            if 'tbn' in model_type.name:
                normal = mathutils.Vector([loop[li].normal[0], loop[li].normal[1], loop[li].normal[2]])
                if export_context.flip_axis == True: normal.rotate(conversion)
                vert.extend([round(normal.x, export_context.precision), round(normal.y, export_context.precision), round(normal.z, export_context.precision)])
            if 'Rigged' in model_type.name:
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
                for bi in range(bones_per_vertex): vert.append(round(vb[bi][1], export_context.precision))
            
            vert_key = str(vert)
            if vert_key not in vert_dict:
                vert_dict[vert_key] = next_index
                
                if export_context.model_buffer_format == 'single_buffer':
                    vertices.extend(vert)
                elif export_context.model_buffer_format == 'separate_buffers':
                    vertices.extend(vert[0:3])
                    next_vert_index = 3
                    if 'uv' in model_type.name:
                        uvs.extend(vert[next_vert_index:next_vert_index+2])
                        next_vert_index += 2
                    if 'tbn' in model_type.name:
                        normals.extend(vert[next_vert_index:next_vert_index+3])
                        next_vert_index += 3
                    if 'Rigged' in model_type.name:
                        weights.extend(vert[next_vert_index:next_vert_index+bones_per_vertex])
                        next_vert_index += bones_per_vertex
                        weight_indices.extend(vert[next_vert_index:next_vert_index+bones_per_vertex])
                        next_vert_index += bones_per_vertex
                else: raise Error('Invalid vertex buffer format.')
                
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

    if export_context.model_buffer_format == 'single_buffer':
        new_model = Model(model_type, vertex_stride, len(vert_dict), vertices, len(indices), indices)
    elif export_context.model_buffer_format == 'separate_buffers':
        new_model = Model(model_type, vertex_stride, len(vert_dict), vertices, len(indices), indices, uvs, normals, weights, weight_indices)
    else: raise Error('Invalid vertex buffer format.')
    
    if export_context.logging == True: new_model.log(export_context.model_buffer_format)
    return new_model


#######################################################################################################################
#                                                    File Writing                                                     #
#######################################################################################################################

def write_to_animation_file(export_context, file_path, frame_range, anim_frames):
    file_path = file_path + ".anim"
    if export_context.logging == True: print('File save location: %s' % file_path)
    
    if export_context.file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and export_context.byte_order != 'little') or (sys.byteorder == 'big' and export_context.byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if export_context.logging == True: print("Writing binary animation data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, export_context.byte_order, signed=True))
        range_byte_array = numpy.array(frame_range[0:2], 'float32')
        if swap_bytes == True: range_byte_array.byteswap()
        range_byte_array.tofile(f)
        f.write(len(anim_frames).to_bytes(4, export_context.byte_order, signed=True))
        f.write(len(anim_frames[0].bone_curves).to_bytes(4, export_context.byte_order, signed=True))
        for anim_frame in anim_frames:
            time_byte_array = numpy.array([float(anim_frame.timestamp)], 'float32')
            if swap_bytes == True: time_byte_array.byteswap()
            time_byte_array.tofile(f)
            for bone_curve in anim_frame.bone_curves:
                f.write(bone_curve.index.to_bytes(4, export_context.byte_order, signed=True))
                location_byte_array = numpy.array(bone_curve.location, 'float32')
                if swap_bytes == True: location_byte_array.byteswap()
                location_byte_array.tofile(f)
                scale_byte_array = numpy.array(bone_curve.scale, 'float32')
                if swap_bytes == True: scale_byte_array.byteswap()
                scale_byte_array.tofile(f)
                quaternion_byte_array = numpy.array(bone_curve.quaternion, 'float32')
                if swap_bytes == True: quaternion_byte_array.byteswap()
                quaternion_byte_array.tofile(f)
    
        f.close()
    elif export_context.file_format == 'ASCII':
        if export_context.logging == True: print("Writing ASCII animation data to file...")
        f = open(file_path, 'w')
        
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('frng %d %d\n' % (frame_range[0], frame_range[1]))
        f.write('fcnt %d\n' % len(anim_frames))
        f.write('bcnt %d\n' % len(anim_frames[0].bone_curves))
        for anim_frame in anim_frames:
            f.write('fnum %d\n' % anim_frame.frame)
            for bone_curve in anim_frame.bone_curves:
                f.write('indx %d\n' % bone_curve.index)
                f.write('loct %f %f %f\n' % (bone_curve.location[0], bone_curve.location[1], bone_curve.location[2]))
                f.write('scal %f %f %f\n' % (bone_curve.scale[0], bone_curve.scale[1], bone_curve.scale[2]))
                f.write('quat %f %f %f %f\n' % (bone_curve.quaternion[0], bone_curve.quaternion[1], bone_curve.quaternion[2], bone_curve.quaternion[3]))
        
        f.close()
    else: raise Error('Invalid file format.')
    

def write_to_armature_file(export_context, file_path, bones_dict):
    file_path = file_path + ".armt"
    if export_context.logging == True: print('File save location: %s' % file_path)
    
    if export_context.file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and export_context.byte_order != 'little') or (sys.byteorder == 'big' and export_context.byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if export_context.logging == True: print("Writing binary armature data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, export_context.byte_order, signed=True))
        f.write(len(bones_dict).to_bytes(4, export_context.byte_order, signed=True))
        for key in bones_dict:
            bone = bones_dict[key]
            f.write(bone.index.to_bytes(4, export_context.byte_order, signed=True))
            f.write(len(bone.name).to_bytes(4, export_context.byte_order, signed=True))
            f.write(bone.name.encode('utf-8'))
            if export_context.armt_matrix == 'local_bind_transform':
                bone_matrix = matrix_to_float16(bone.local_bind_transform.transposed(), export_context.precision)
            elif export_context.armt_matrix == 'inverse_bind_transform':
                bone_matrix = matrix_to_float16(bone.inverse_bind_transform.transposed(), export_context.precision)
            else: raise Error('Invalid bone matrix transformation.')
            matrix_byte_array = numpy.array(bone_matrix, 'float32')
            if swap_bytes == True: matrix_byte_array.byteswap()
            matrix_byte_array.tofile(f)
            children_indices = bone.get_children_indices()
            f.write(len(children_indices).to_bytes(4, export_context.byte_order, signed=True))
            indices_byte_array = numpy.array(children_indices, 'int_')
            if swap_bytes == True: indices_byte_array.byteswap()
            indices_byte_array.tofile(f)
        
        f.close()
    elif export_context.file_format == 'ASCII':
        if export_context.logging == True: print("Writing ASCII armature data to file...")
        f = open(file_path, 'w')
        
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('bcnt %d\n' % len(bones_dict))
        for key in bones_dict:
            bone = bones_dict[key]
            f.write('indx %d\n' % bone.index)
            f.write('nsiz %d\n' % len(bone.name))
            f.write('narr %s\n' % bone.name)
            f.write('matx')
            if export_context.armt_matrix == 'local_bind_transform':
                bone_matrix = matrix_to_float16(bone.local_bind_transform, export_context.precision)
            elif export_context.armt_matrix == 'inverse_bind_transform':
                bone_matrix = matrix_to_float16(bone.inverse_bind_transform, export_context.precision)
            else: raise Error('Invalid bone matrix transformation.')
            for num in bone_matrix: f.write(' %f' % num)
            f.write('\n')
            children_indices = bone.get_children_indices()
            f.write('isiz %d\n' % len(children_indices))
            f.write('iarr')
            for indx in children_indices: f.write(' %d' % indx)
            f.write('\n')
        
        f.close()
    else: raise Error('Invalid file format.')
    

def write_to_model_file(export_context, file_path, model):
    file_path = file_path + ".model"
    if export_context.logging == True: print('File save location: %s' % file_path)
    
    if export_context.file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and export_context.byte_order != 'little') or (sys.byteorder == 'big' and export_context.byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if export_context.logging == True: print("Writing binary model data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, export_context.byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, export_context.byte_order, signed=True))
        f.write(model.type.value.to_bytes(4, export_context.byte_order, signed=True))
        f.write(model.vertex_stride.to_bytes(4, export_context.byte_order, signed=True))
        f.write(model.vertex_count.to_bytes(4, export_context.byte_order, signed=True))
        if export_context.model_buffer_format == 'single_buffer':
            vertices_byte_array = numpy.array(model.vertices, 'float32')
            if swap_bytes == True: vertices_byte_array.byteswap()
            vertices_byte_array.tofile(f)
        elif export_context.model_buffer_format == 'separate_buffers':
            vertices_byte_array = numpy.array(model.vertices, 'float32')
            if swap_bytes == True: vertices_byte_array.byteswap()
            vertices_byte_array.tofile(f)
            if 'uv' in model.type.name:
                uvs_byte_array = numpy.array(model.uvs, 'float32')
                if swap_bytes == True: uvs_byte_array.byteswap()
                uvs_byte_array.tofile(f)
            if 'tbn' in model.type.name:
                normals_byte_array = numpy.array(model.normals, 'float32')
                if swap_bytes == True: normals_byte_array.byteswap()
                normals_byte_array.tofile(f)
            if 'Rigged' in model.type.name:
                weight_indices_byte_array = numpy.array(model.weight_indices, 'int_')
                if swap_bytes == True: weight_indices_byte_array.byteswap()
                weight_indices_byte_array.tofile(f)
                weights_byte_array = numpy.array(model.weights, 'float32')
                if swap_bytes == True: weights_byte_array.byteswap()
                weights_byte_array.tofile(f)
        else: raise Error('Invalid vertex buffer format.')
        f.write(model.material_count.to_bytes(4, export_context.byte_order, signed=True))
        for i in range(model.material_count):
            f.write(len(model.indices_list[i]).to_bytes(4, export_context.byte_order, signed=True))
            indices_byte_array = numpy.array(model.indices_list[i], 'int_')
            if swap_bytes == True: indices_byte_array.byteswap()
            indices_byte_array.tofile(f)
           
        f.close()
    elif export_context.file_format == 'ASCII':
        if export_context.logging == True: print("Writing ASCII model data to file...")
        f = open(file_path, 'w')
            
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('mtyp %d\n' % model.type.value)
        f.write('vstr %d\n' % model.vertex_stride)
        f.write('vcnt %d\n' % model.vertex_count)
        if export_context.model_buffer_format == 'single_buffer':
            f.write('varr')
            for vert in model.vertices: f.write(' %f' % vert)
            f.write('\n')
        elif export_context.model_buffer_format == 'separate_buffers':
            f.write('varr')
            for vert in model.vertices: f.write(' %f' % vert)
            f.write('\n')
            if 'uv' in model.type.name:
                f.write('tarr')
                for uv in model.uvs: f.write(' %f' % uv)
                f.write('\n')
            if 'tbn' in model.type.name:
                f.write('narr')
                for normal in model.normals: f.write(' %f' % normal)
                f.write('\n')
            if 'Rigged' in model.type.name:
                f.write('wiar')
                for weight_index in model.weight_indices: f.write(' %d' % weight_index)
                f.write('\n')
                f.write('warr')
                for weight in model.weights: f.write(' %f' % weight)
                f.write('\n')
        f.write('matc %d\n' % model.material_count)
        for i in range(model.material_count):
            f.write('isiz %d\n' % len(model.indices_list[i]))
            f.write('iarr')
            for indx in model.indices_list[i]: f.write(' %d' % indx)
            f.write('\n')
            
        f.close()      
    else: raise Error('Invalid file format.')


#######################################################################################################################
#                                                       Other                                                         #
#######################################################################################################################

def make_objects_list(export_context):
    if export_context.logging == True: print('Fetching objects list...')
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
    if export_context.object_selection == 'selected_only':
        recursive = False
        selected_objects = bpy.context.selected_objects
    elif export_context.object_selection == 'selected_children':
        recursive = True
        selected_objects = bpy.context.selected_objects
    elif export_context.object_selection == 'all_objects':
        recursive = True
        selected_objects = bpy.context.visible_objects
    else: raise Error('Invalid selection.')

    for object in selected_objects:
        if object == None:  continue 
        recursive_search(object, recursive)
    
    if len(mesh_list) == 0: 
        raise Error('Object selection(s) have no mesh to export.')
    if export_context.logging == True: 
        print(' > Mesh List:', mesh_list)
        print(' > Armature List:', armature_list)
    
    return mesh_list, armature_list


def get_model_type(export_context):
    if export_context.logging == True: print('Finding Model Type...')
    type = None
    if export_context.include_uvs == True:
        if export_context.include_normals == True:
            if export_context.include_bones == True: type = ModelType.xyzuvtbnRigged
            else: type = ModelType.xyzuvtbn
        else:
            if export_context.include_bones == True: type = ModelType.xyzuvRigged
            else: type = ModelType.xyzuv
    else:
        if export_context.include_normals == True:
            if export_context.include_bones == True: type = ModelType.xyztbnRigged
            else: type = ModelType.xyztbn
        else:
            if export_context.include_bones == True: type = ModelType.xyzRigged
            else: type = ModelType.xyz
            
    if export_context.logging == True: print(' > Model Type: %s' % type.name)
    return type


#######################################################################################################################
#                                                   Exporter Setup                                                    #
#######################################################################################################################

def execute_exporter(export_context, filepath):
    print('\nEXPORTER STARTED')
    export_context.log()
    
    try:
        # Set mode to Object
        bpy.ops.object.mode_set(mode='OBJECT')
        
        if export_context.include_model == False and export_context.include_armt == False and export_context.include_anim == False:
            raise Error('Nothing to export. Select object to export in the "Includes" box.')
        
        # Fetch all the objects to export
        object_list, armature_list = make_objects_list(export_context)
        
        # Find ModelType to export
        if export_context.include_model == True:
            model_type = get_model_type(export_context)
            
        # Process all Armature data for use in processing other data
        if export_context.include_bones == True or export_context.include_armt == True or export_context.include_anim == True:
            armature_dict = {}
            for armature in armature_list:
                armature_dict[armature.name] = process_armature_data(export_context, armature.data)
        
        # Write all Armature data to file
        if export_context.include_armt == True:
            for key in armature_dict:
                write_to_armature_file(export_context, filepath, armature_dict[key])
            print('ARMATURE EXPORT FINISHED')
        
        # Process and export all Animation data
        if export_context.include_anim == True:
            for armature in armature_list:
                frame_range, anim_frames = process_animation_data(export_context, armature, armature_dict[armature.name])
                write_to_animation_file(export_context, filepath, frame_range, anim_frames)
            print('ANIMATION EXPORT FINISHED')
           
        # Process and export all model data 
        if export_context.include_model == True:
            for object in object_list:
                # Fetch Object's Armature data (aka. bones_dict) for current object
                bones_dict = None
                if export_context.include_bones == True:
                    for modifier in object.modifiers:
                        if modifier.type == 'ARMATURE':
                            bones_dict = armature_dict[modifier.object.name]
                        break
                    if bones_dict == None: raise Error('Object "%s" missing armature.' % object.name)
                
                # Fetch and process Object's Model data
                model = process_model_data(export_context, object, model_type, bones_dict)
                # Modify filepath if necessary
                model_filepath = filepath if len(object_list) == 1 else filepath + '_' + object.name.replace(':', '_')
                # Write Model data to file
                write_to_model_file(export_context, model_filepath, model)
            print('MODEL EXPORT FINISHED')    
        
    except Error as e: 
        print('Error:', e)
        
    print('EXPORTER FINISHED')
    return {'FINISHED'}


# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, BoolVectorProperty, IntProperty, EnumProperty
from bpy.types import Operator, Panel

class OpenGLExporter(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "opengl_exporter.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export"
    filename_ext = "" # ExportHelper mixin class uses this

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
    file_format: EnumProperty(
        name="Format",
        description="Choose file format",
        items=(
            ('ASCII',  "ASCII",  "Text file format"),
            ('Binary', "Binary", "Binary file format"),
        ),
        default='Binary',
    )
    byte_order: EnumProperty(
        name="Byte Order",
        description="Choose byte order for binary file",
        items=(
            ('little',  "Little-Endian",  "Little-Endian byte order"),
            ('big', "Big-Endian", "Big-Endian byte order"),
        ),
        default='little',
    )
    precision: IntProperty(
        name="Precision",
        description="Rounding precision applied to all float values",
        default=6, min=2, max=8
    )
    flip_axis: BoolProperty(
        name="Y-Axis as Up",
        description="Flips the axis so that the Y-Axis is pointing up.\nCurrently not available",
        default=True,
    )
    logging: BoolProperty(
        name="Log to console",
        description="Logs all data and calls to console",
        default=False,
    )
    
    # Includes Properties
    model_bool: BoolProperty(
        name="Model",
        description="Writes model data and exports it in .model file",
        default=True,
    )
    armt_bool: BoolProperty(
        name="Armature",
        description="Writes bone data in a hierarchy list and exports it in .armt file",
        default=True,
    )
    anim_bool: BoolProperty(
        name="Animation",
        description="Writes animation data and exports it in .anim file",
        default=True,
    )
    
    # Model Settings Properties
    buffer_format: EnumProperty(
        name="Vertex Buffer",
        description="Choose the vertex buffer format",
        items=(
            ('single_buffer',  "Single Buffer",  "All vertex data will be packed into one buffer.\nDoes not work for rigging data"),
            ('separate_buffers', "Separate Buffers", "Vertex data will be packed into separate buffers.\nUseful for use with VAO in OpenGL"),
        ),
        default='separate_buffers',
    )
    uv_bool: BoolProperty(
        name="UVs",
        description="Writes UV data in vertex buffer",
        default=True,
    )
    normals_bool: BoolProperty(
        name="Normals",
        description="Writes normal data in vertex buffer",
        default=True,
    )
    rigging_bool: BoolProperty(
        name="Rigging",
        description='Writes bone weight and index in vertex buffer.\nCan only be exported if vertex buffer format is set to "Separate Buffers"',
        default=True,
    ) 
    
    # Armature Settings
    armt_matrix: EnumProperty(
        name="Matrix",
        description="Choose matrix transformation to be exported for each bone",
        items=(
            ('local_bind_transform',  "Local Bind Transform",  "Bone space rest pose matrix transformation"),
            ('inverse_bind_transform', "Inverse Bind Transform", "Inversed world space rest pose matrix transformation"),
        ),
        default='inverse_bind_transform',
    )
    
    # Animation Settings
    anim_time_format: EnumProperty(
        name="Time Format",
        description="Choose the way time is represented",
        items=(
            ('seconds', "Seconds", "Timestamp is in seconds (1 second is 24 frames)"),
            ('frames',  "Frames",  "Timestamp is in frames"),
        ),
        default='seconds',
    )
    anim_export_frames: EnumProperty(
        name="Frames",
        description="Choose frames to export",
        items=(
            ('action_pose_markers',  "Action Pose Markers",  "Export only frames marked with an ActionPoseMarker"),
            ('keyframe_points', "All Keyframe Points", "Export all frames that have a Keyframe point"),
            ('interval', "Specified Interval", "Export frames at the specified interval"),
        ),
        default='keyframe_points',
    )
    anim_frame_interval: IntProperty(
        name="Interval",
        description='Interval between each frames to be exported.\nFrames must be set to "Specified Interval"',
        default=10, min=1
    )
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        general_box = layout.box()
        general_box.label(text='General Settings:')
        general_box.prop(self, 'selection')
        general_box.prop(self, 'file_format')
        general_box_byte_order = general_box.column()
        general_box_byte_order.prop(self, 'byte_order')
        general_box_byte_order.enabled = True if self.file_format == 'Binary' else False 
        general_box.prop(self, 'precision')
        general_box.prop(self, 'flip_axis')
        general_box.prop(self, 'logging')
        general_includes_box = general_box.box()
        general_includes_box.label(text='Includes:')
        general_includes_box.prop(self, 'model_bool')
        general_includes_box.prop(self, 'armt_bool')
        general_includes_box.prop(self, 'anim_bool')
        
        model_box = layout.box()
        model_box.label(text='Model Settings:')
        model_box.prop(self, 'buffer_format')
        model_includes_box = model_box.box()
        model_includes_box.label(text='Includes:')
        model_includes_box.prop(self, 'uv_bool')
        model_includes_box.prop(self, 'normals_bool')
        model_includes_box_rigging = model_includes_box.column()
        model_includes_box_rigging.prop(self, 'rigging_bool')
        model_includes_box_rigging.enabled = True if self.buffer_format == 'separate_buffers' else False 
        model_box.enabled = self.model_bool
        
        armt_box = layout.box()
        armt_box.label(text='Armature Settings:')
        armt_box.prop(self, 'armt_matrix')
        armt_box.enabled = self.armt_bool
        
        anim_box = layout.box()
        anim_box.label(text='Animation Settings:')
        anim_box.prop(self, 'anim_time_format')
        anim_box.prop(self, 'anim_export_frames')
        anim_box_interval = anim_box.column()
        anim_box_interval.prop(self, 'anim_frame_interval')
        anim_box_interval.enabled = True if self.anim_export_frames == 'interval' else False 
        anim_box.enabled = self.anim_bool

    def execute(self, context):
        export_context = ExportContext()
        export_context.object_selection = self.selection
        export_context.file_format = self.file_format
        export_context.byte_order = self.byte_order
        export_context.precision = self.precision
        export_context.flip_axis = self.flip_axis
        export_context.logging = self.logging
        export_context.include_model = self.model_bool
        export_context.include_armt = self.armt_bool
        export_context.include_anim = self.anim_bool
        export_context.model_buffer_format = self.buffer_format
        export_context.include_uvs = self.uv_bool
        export_context.include_normals = self.normals_bool
        export_context.include_bones = self.rigging_bool if self.buffer_format == 'separate_buffers' else False
        export_context.armt_matrix = self.armt_matrix
        export_context.anim_time_format = self.anim_time_format
        export_context.anim_export_frames = self.anim_export_frames
        export_context.anim_frame_interval = self.anim_frame_interval
        return execute_exporter(export_context, self.filepath.replace(self.filename_ext, ''))
        
    
# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(OpenGLExporter.bl_idname, text="OpenGL Exporter (.model, .amrt, .anim)")


def register():
    bpy.utils.register_class(OpenGLExporter)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(OpenGLExporter)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.opengl_exporter.some_data('INVOKE_DEFAULT')
    
