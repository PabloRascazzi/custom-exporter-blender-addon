import bpy
import sys
import numpy

bl_info = {
    "name": "Model Exporter Add-on",
    "description": "Exports model data with vertices and indices in a proper format for OpenGL.",
    "author": "Pablo Rascazzi",
    "version": (0, 1),
    "blender": (2, 92, 0),
    "location": "File > Export > Model Exporter (.mdl)",
    "category": "Import-Export"
}

logging = False

def process_model_data(object, model_type):
    if logging == True: print("Processing object data...")
    
    if model_type == 'xyz': vertices_stride = 3
    elif model_type == 'xyzuv': vertices_stride = 5
    elif model_type == 'xyzuvtbn': vertices_stride = 8
    else: sys.exit('Error: Invalid model type.')
    
    mesh = object.data
    loop = mesh.loops
    
    if vertices_stride >= 5: uv_layer = mesh.uv_layers.active.data
    if vertices_stride >= 8: mesh.calc_tangents()
    
    vert_dict = {}
    vertices = []
    indices = [[]]
    
    next_index = 0
    for face in mesh.polygons:
        tmp_indices = []

        # processing vertices per faces...
        for li in face.loop_indices:
            vert = []
            vert.append(mesh.vertices[loop[li].vertex_index].co.x)
            vert.append(mesh.vertices[loop[li].vertex_index].co.y)
            vert.append(mesh.vertices[loop[li].vertex_index].co.z)
            if vertices_stride >= 5:
                vert.append(uv_layer[li].uv.x)
                vert.append(uv_layer[li].uv.y)
            if vertices_stride >= 8:
                vert.extend(loop[li].normal)
            
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

    vertices_size = int(len(vertices)/vertices_stride)
    indices_size = len(indices)
    
    return vertices_stride, vertices_size, vertices, indices_size, indices

def write_model_data(context, filepath, model_type, file_format, separate_object):
    vertices_stride, vertices_size, vertices, indices_size, indices = process_model_data(bpy.context.object, model_type)

    if logging == True: # Logs for debugging
        print('Exporter Version: %d.%d' % (bl_info['version'][0], bl_info['version'][1]))
        print('Vertices Stride: %d' % vertices_stride)
        print('Vertices Size: %d' % vertices_size)
        print('Vertices Array:', vertices)
        print('Material Count: %d' % indices_size)
        for i in range(indices_size):
            print('Indices[%d] Size:' % i,len(indices[i]))
            print('Indices[%d]:' % i, indices[i])
    
    if file_format == 'Binary':
        if logging == True: print("Writing binary data to file...")
        f = open(filepath, 'wb')
        
        f.write(bl_info['version'][0].to_bytes(4, sys.byteorder, signed=True))
        f.write(bl_info['version'][1].to_bytes(4, sys.byteorder, signed=True))
        f.write(vertices_stride.to_bytes(4, sys.byteorder, signed=True))
        f.write(vertices_size.to_bytes(4, sys.byteorder, signed=True))
        vertices_byte_array = numpy.array(vertices, 'float32')
        vertices_byte_array.tofile(f)
        f.write(indices_size.to_bytes(4, sys.byteorder, signed=True))
        for i in range(indices_size):
            f.write(len(indices[i]).to_bytes(4, sys.byteorder, signed=True))
            indices_byte_array = numpy.array(indices[i], 'int_')
            indices_byte_array.tofile(f)
        
        f.close()
    elif file_format == 'ASCII':
        if logging == True: print("Writing ASCII data to file...")
        f = open(filepath, 'w')
        
        f.write('vers %d %d\n' % (bl_info['version'][0], bl_info['version'][1]))
        f.write('vstr %d\n' % vertices_stride)
        f.write('vsiz %d\n' % vertices_size)
        f.write('varr')
        for vert in vertices:
            f.write(' %f' % vert)
        f.write('\n')
        f.write('matc %d\n' % indices_size)
        for i in range(indices_size):
            f.write('isiz %d\n' % len(indices[i]))
            f.write('iarr')
            for indx in indices[i]:
                f.write(' %d' % indx)
            f.write('\n')
        
        f.close()
        
    else:
        sys.exit('Error: Invalid file format.')

    print('MODEL EXPORT FINISHED')
    return {'FINISHED'}


# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator


class ExportModel(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_model.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Model"

    # ExportHelper mixin class uses this
    filename_ext = ".mdl"

    filter_glob: StringProperty(
        default="*.model",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    type: EnumProperty(
        name="Model Type",
        description="Choose model type",
        items=(
            ('xyz', "xyz", "Export only position coordinates"),
            ('xyzuv', "xyzuv", "Export position and texture coordinates"),
            ('xyzuvtbn', "xyzuvtbn", "Export position, texture, and normals coordinates"),
        ),
        default='xyzuvtbn',
    )
    
    format: EnumProperty(
        name="Format",
        description="Choose file format",
        items=(
            ('ASCII', "ASCII", "Text file format"),
            ('Binary', "Binary", "Binary file format"),
        ),
        default='Binary',
    )
    
    use_setting: BoolProperty(
        name="Separate object",
        description="Separate all objects into different files",
        default=True,
    )

    def execute(self, context):
        return write_model_data(context, self.filepath, self.type, self.format, self.use_setting)


# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(ExportModel.bl_idname, text="Model Exporter (.mdl)")


def register():
    bpy.utils.register_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportModel)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.export_test.some_data('INVOKE_DEFAULT')
