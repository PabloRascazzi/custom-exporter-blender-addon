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
    "version": (0, 7, 1),
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


class BoneCurve():
    def __init__(self):
        self.index = None
        self.name = None
        self.location = [0,0,0]
        self.scale = [0,0,0]
        self.quaternion = [0,0,0,0]
        
    def log(self):
        print(' > Bone Name: %s' % self.name)
        print(' == > Bone Index: %d' % self.index)
        print(' == > Location:   ', self.location)
        print(' == > Scale:      ', self.scale)
        print(' == > Quaternion: ', self.quaternion)
    
    
class Animframe():
    def __init__(self):
        self.timestamp = None
        self.bone_curves = []
        
    def log(self):
        print(' > Timestamp: %f' % self.timestamp)
        print(' > Bone Curves:')
        for curve in self.bone_curves: curve.log()
    

#######################################################################################################################
#                                                     Utilities                                                       #
#######################################################################################################################

def matrix_to_float16(matrix, precision):
    float16 = []
    float16.extend([round(matrix[0][0], precision), round(matrix[0][1], precision), round(matrix[0][2], precision), round(matrix[0][3], precision)])
    float16.extend([round(matrix[1][0], precision), round(matrix[1][1], precision), round(matrix[1][2], precision), round(matrix[1][3], precision)])
    float16.extend([round(matrix[2][0], precision), round(matrix[2][1], precision), round(matrix[2][2], precision), round(matrix[2][3], precision)])
    float16.extend([round(matrix[3][0], precision), round(matrix[3][1], precision), round(matrix[3][2], precision), round(matrix[3][3], precision)])
    return float16
    

#######################################################################################################################
#                                                  Data Processing                                                    #
#######################################################################################################################

def process_armature_data(armature, flip_axis, precision, logging):
    if logging == True: print("Processing armature data...")   
    bones_dict = {} # dict of bones with bone names as key (bone.name == vertex_group.name)
    
    root_bone = None
    next_index = 0
    
    # Create BoneInfo and calculate local_bind_transform from armature data
    for bone in armature.bones:
        new_bone = BoneInfo()
        new_bone.index = next_index
        new_bone.name = bone.name
        
        if bone.parent == None: 
            new_bone.local_bind_transform = bone.matrix_local
            root_bone = new_bone
        else: 
            new_bone.local_bind_transform = bone.parent.matrix_local.inverted() @ bone.matrix_local.copy()
            bones_dict[bone.parent.name].children.append(new_bone)
        
        bones_dict[bone.name] = new_bone
        next_index += 1
        
    # Recursively call calc_inverse_bind_transform() to calculate all inverse_bind_transform
    root_bone.calc_inverse_bind_transform(flip_axis, mathutils.Matrix().Identity(4))

    # Log BoneInfo dictionary
    if logging == True:  
        for key in bones_dict: bones_dict[key].log(precision)
            
    return bones_dict


def process_animation_data(armature, bones_dict, flip_axis, export_frames, frame_interval, time_format, precision, logging):
    if logging == True: print("Processing animation data...")
    action = armature.animation_data.action
    fps = bpy.context.scene.render.fps
    
    if time_format == 'frames': frame_range = [float(action.frame_range[0]), float(action.frame_range[1])]
    elif time_format == 'seconds': frame_range = [round(action.frame_range[0]/fps, 8), round(action.frame_range[1]/fps, 8)]
    else: raise Error('Invalid animation time format.')
    
    frame_times = []
    anim_frames = []

    if export_frames == 'action_pose_markers':
        for marker in action.pose_markers: 
            frame_times.append(marker.frame)
    elif export_frames == 'keyframe_points':
        for fcurve in action.fcurves:
            frame_times.extend([int(time.co[0]) for time in fcurve.keyframe_points if int(time.co[0]) not in frame_times]) 
    elif export_frames == 'interval':
        frame_times = list(range(int(action.frame_range[0]), int(action.frame_range[1]+1), frame_interval))
    else: raise Error('Invalid animation export frames.')
    frame_times.sort()
    
    if logging == True: 
        print('Frame Range:', frame_range)
        print('Frame Times:', frame_times)
    
    for frame in frame_times:
        new_frame = Animframe()
        if time_format == 'frames': new_frame.timestamp = frame
        elif time_format == 'seconds': new_frame.timestamp = (round(frame/fps, 8))
        else: raise Error('Invalid animation time format.')
        
        for group in action.groups:
            if group.name not in bones_dict:
                print('Warning: Non-bone FCurves not supported.') 
                continue
            
            new_curve = BoneCurve()
            new_curve.name = group.name
            new_curve.index = bones_dict[group.name].index
            quaternion_list = [1,0,0,0]
            
            for fcurve in group.channels:
                if   fcurve.data_path.endswith('location'): 
                    new_curve.location[fcurve.array_index] = round(fcurve.evaluate(frame), precision)
                elif fcurve.data_path.endswith('scale'): 
                    new_curve.scale[fcurve.array_index] = round(fcurve.evaluate(frame), precision)
                elif fcurve.data_path.endswith('rotation_quaternion'):
                    quaternion_list[fcurve.array_index] = fcurve.evaluate(frame)
                    
            quaternion = mathutils.Quaternion(quaternion_list)
            new_curve.quaternion = [round(quaternion.x, precision), round(quaternion.y, precision), round(quaternion.z, precision), round(quaternion.w, precision)]
            
            new_frame.bone_curves.append(new_curve)
        anim_frames.append(new_frame)
            
    if logging == True: 
        print('Animation Frames:')
        for anim_frame in anim_frames: anim_frame.log()
        
    return frame_range, anim_frames


def process_model_data(object, model_type, bones_dict, buffer_format, flip_axis, precision, logging):
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
            if flip_axis == True: position.rotate(conversion)
            vert.extend([round(position.x, precision), round(position.y, precision), round(position.z, precision)])
            if 'uv' in model_type.name:
                vert.append(round(uv_layer[li].uv.x, precision))
                vert.append(round(uv_layer[li].uv.y, precision))
            if 'tbn' in model_type.name:
                normal = mathutils.Vector([loop[li].normal[0], loop[li].normal[1], loop[li].normal[2]])
                if flip_axis == True: normal.rotate(conversion)
                vert.extend([round(normal.x, precision), round(normal.y, precision), round(normal.z, precision)])
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
                for bi in range(bones_per_vertex): vert.append(round(vb[bi][1], precision))
            
            vert_key = str(vert)
            if vert_key not in vert_dict:
                vert_dict[vert_key] = next_index
                
                if buffer_format == 'single_buffer':
                    vertices.extend(vert)
                elif buffer_format == 'separate_buffers':
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

    if buffer_format == 'single_buffer':
        new_model = Model(model_type, vertex_stride, len(vert_dict), vertices, len(indices), indices)
    elif buffer_format == 'separate_buffers':
        new_model = Model(model_type, vertex_stride, len(vert_dict), vertices, len(indices), indices, uvs, normals, weights, weight_indices)
    else: raise Error('Invalid vertex buffer format.')
    return new_model


#######################################################################################################################
#                                                    File Writing                                                     #
#######################################################################################################################

def write_to_animation_file(file_path, file_format, byte_order, frame_range, anim_frames, logging):
    file_path = file_path + ".anim"
    if logging == True: print('File save location: %s' % file_path)
    
    if file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and byte_order != 'little') or (sys.byteorder == 'big' and byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if logging == True: print("Writing binary animation data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, byte_order, signed=True))
        range_byte_array = numpy.array(frame_range[0:2], 'float32')
        if swap_bytes == True: range_byte_array.byteswap()
        range_byte_array.tofile(f)
        f.write(len(anim_frames).to_bytes(4, byte_order, signed=True))
        f.write(len(anim_frames[0].bone_curves).to_bytes(4, byte_order, signed=True))
        for anim_frame in anim_frames:
            time_byte_array = numpy.array([float(anim_frame.timestamp)], 'float32')
            if swap_bytes == True: time_byte_array.byteswap()
            time_byte_array.tofile(f)
            for bone_curve in anim_frame.bone_curves:
                f.write(bone_curve.index.to_bytes(4, byte_order, signed=True))
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
    elif file_format == 'ASCII':
        if logging == True: print("Writing ASCII animation data to file...")
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
    

def write_to_armature_file(file_path, file_format, byte_order, bones_dict, armt_matrix, precision, logging):
    file_path = file_path + ".armt"
    if logging == True: print('File save location: %s' % file_path)
    
    if file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and byte_order != 'little') or (sys.byteorder == 'big' and byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if logging == True: print("Writing binary armature data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, byte_order, signed=True))
        f.write(len(bones_dict).to_bytes(4, byte_order, signed=True))
        for key in bones_dict:
            bone = bones_dict[key]
            f.write(bone.index.to_bytes(4, byte_order, signed=True))
            f.write(len(bone.name).to_bytes(4, byte_order, signed=True))
            f.write(bone.name.encode('utf-8'))
            if armt_matrix == 'local_bind_transform':
                bone_matrix = matrix_to_float16(bone.local_bind_transform.transposed(), precision)
            elif armt_matrix == 'inverse_bind_transform':
                bone_matrix = matrix_to_float16(bone.inverse_bind_transform.transposed(), precision)
            else: raise Error('Invalid bone matrix transformation.')
            matrix_byte_array = numpy.array(bone_matrix, 'float32')
            if swap_bytes == True: matrix_byte_array.byteswap()
            matrix_byte_array.tofile(f)
            children_indices = bone.get_children_indices()
            f.write(len(children_indices).to_bytes(4, byte_order, signed=True))
            indices_byte_array = numpy.array(children_indices, 'int_')
            if swap_bytes == True: indices_byte_array.byteswap()
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
            if armt_matrix == 'local_bind_transform':
                bone_matrix = matrix_to_float16(bone.local_bind_transform, precision)
            elif armt_matrix == 'inverse_bind_transform':
                bone_matrix = matrix_to_float16(bone.inverse_bind_transform, precision)
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
    

def write_to_model_file(file_path, file_format, byte_order, model, buffer_format, logging):
    file_path = file_path + ".model"
    if logging == True: print('File save location: %s' % file_path)
    
    if file_format == 'Binary':
        swap_bytes = False
        if (sys.byteorder == 'little' and byte_order != 'little') or (sys.byteorder == 'big' and byte_order != 'big'): swap_bytes = True
        elif sys.byteorder != 'little' and sys.byteorder != 'big': raise Error('Invalid byte order.')
        
        if logging == True: print("Writing binary model data to file...")
        f = open(file_path, 'wb')
        
        f.write(identifier.to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][0].to_bytes(4, byte_order, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, byte_order, signed=True))
        f.write(model.type.value.to_bytes(4, byte_order, signed=True))
        f.write(model.vertex_stride.to_bytes(4, byte_order, signed=True))
        f.write(model.vertex_count.to_bytes(4, byte_order, signed=True))
        if buffer_format == 'single_buffer':
            vertices_byte_array = numpy.array(model.vertices, 'float32')
            if swap_bytes == True: vertices_byte_array.byteswap()
            vertices_byte_array.tofile(f)
        elif buffer_format == 'separate_buffers':
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
        f.write(model.material_count.to_bytes(4, sys.byteorder, signed=True))
        for i in range(model.material_count):
            f.write(len(model.indices_list[i]).to_bytes(4, byte_order, signed=True))
            indices_byte_array = numpy.array(model.indices_list[i], 'int_')
            if swap_bytes == True: indices_byte_array.byteswap()
            indices_byte_array.tofile(f)
           
        f.close()
    elif file_format == 'ASCII':
        if logging == True: print("Writing ASCII model data to file...")
        f = open(file_path, 'w')
            
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('mtyp %d\n' % model.type.value)
        f.write('vstr %d\n' % model.vertex_stride)
        f.write('vcnt %d\n' % model.vertex_count)
        if buffer_format == 'single_buffer':
            f.write('varr')
            for vert in model.vertices: f.write(' %f' % vert)
            f.write('\n')
        elif buffer_format == 'separate_buffers':
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


#######################################################################################################################
#                                                   Exporter Setup                                                    #
#######################################################################################################################

def execute_exporter(self, context):
    print('\nEXPORTER STARTED')
    print('Exporter Version: %d.%d.%d' % (bl_info['version'][0], bl_info['version'][1], bl_info['version'][1]))
    
    try:
        if self.model_bool == False and self.armt_bool == False and self.anim_bool == False:
            raise Error('Nothing to export. Select object to export in the "Includes" box.')
            
        file_path = self.filepath.replace(self.filename_ext, '')
        self.rigging_bool = self.rigging_bool if self.buffer_format == 'separate_buffers' else False
        
        # Fetch all the objects to export
        object_list, armature_list = make_objects_list(self.selection, self.logging)
        
        # Find ModelType to export
        if self.model_bool == True:
            model_type = get_model_type(self.uv_bool, self.normals_bool, self.rigging_bool, self.logging)
            
        # Process all Armature data for use in processing other data
        if self.rigging_bool == True or self.armt_bool == True or self.anim_bool == True:
            armature_dict = {}
            for armature in armature_list:
                armature_dict[armature.name] = process_armature_data(armature.data, self.flip_axis, self.precision, self.logging)
        
        # Write all Armature data to file
        if self.armt_bool == True:
            for key in armature_dict:
                write_to_armature_file(file_path, self.file_format, self.byte_order, armature_dict[key], self.armt_matrix, self.precision, self.logging)
            print('ARMATURE EXPORT FINISHED')
        
        # Process and export all Animation data
        if self.anim_bool == True:
            for armature in armature_list:
                frame_range, anim_frames = process_animation_data(armature, armature_dict[armature.name], self.flip_axis, self.anim_export_frames, self.anim_frame_interval, self.anim_time_format, self.precision, self.logging)
                write_to_animation_file(file_path, self.file_format, self.byte_order, frame_range, anim_frames, self.logging)
            print('ANIMATION EXPORT FINISHED')
           
        # Process and export all model data 
        if self.model_bool == True:
            for object in object_list:
                # Fetch Object's Armature data (aka. bones_dict) for current object
                bones_dict = None
                if self.rigging_bool == True:
                    for modifier in object.modifiers:
                        if modifier.type == 'ARMATURE':
                            bones_dict = armature_dict[modifier.object.name]
                        break
                    if bones_dict == None: raise Error('Object "%s" missing armature.' % object.name)
                
                # Fetch and process Object's Model data
                model = process_model_data(object, model_type, bones_dict, self.buffer_format, self.flip_axis, self.precision, self.logging)
                if self.logging == True: model.log(self.buffer_format)
                
                # Modify file_path if necessary
                model_file_path = file_path if len(object_list) == 1 else file_path + '_' + object.name.replace(':', '_')
                
                # Write Model data to file
                write_to_model_file(model_file_path, self.file_format, self.byte_order, model, self.buffer_format, self.logging)
                if self.logging == True: print('Finished exporting %s...' % object.name)
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
        default=False,
    )
    logging: BoolProperty(
        name="Log to console",
        description="Logs all data and calls to console",
        default=True,
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
        general_box_disabled = general_box.column()
        general_box_disabled.prop(self, 'flip_axis')
        general_box_disabled.enabled = True
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
        return execute_exporter(self, context)
        
    
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
    
