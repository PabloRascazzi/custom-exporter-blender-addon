OpenGL Exporter Version 0.7 (Additional Information)

Patch Notes:
 - Renamed the add-on to OpenGL Exporter
 - Added the option to separate the vertex buffers into multiple buffer (great for use with VAO in OpenGL)
 - Added the option to choose float precisions in the general settings
 - Added the option to export the "Inverse Bind Transform" of the "Local Bind Transform" as the armature's Bone Pose Transform
 - Added the option to change byte order between "Little-Endian" and "Big-Endian"
 - Added an identifier to the FileHeader to determine the byte order (appears as RASC for little-endian, and CSAR for big-endian)
 - Changing animation frames now works
 - Reworked the GUI to be easier to use with all the new options
 - Major bug fixes

Data Structures:

Enum ModelType {
    xyz, xyzuv, xyztbn, xyzuvtbn, xyzRigged, xyzuvRigged, xyztbnRigged, xyzuvtbnRigged
}

struct FileHeader { 
    int identifier; // 4 bytes
    int version[2]; // 8 bytes, Major and Minor version number
}

MODEL File Format (.model) (Separate Vertex Buffers)
 > FileHeader:               12 byte
 > Model Type Enum:          4 byte -> int
 > Vertices Stride:          4 byte -> long
 > Vertex Count:             4 byte -> long
 > Vertices Array:           4 byte -> float[Vertex Count * 3]
 > UVs Array:                4 byte -> float[Vertex Count * 2] (Only if UVs are included)
 > Normals Array:            4 byte -> float[Vertex Count * 3] (Only if Normals are included)
 > Weight Indices Array:     4 byte -> long [Vertex Count * 3] (Only if Rigging is included)
 > WeightsArray:             4 byte -> float[Vertex Count * 3] (Only if Rigging is included)
 > Material Count:           4 byte -> long
 > Foreach Material Count:
 === > Indices Size:         4 byte -> long
 === > Indices Array:        4 byte -> long[Indices Size]

MODEL File Format (.model) (Single Vertex Buffer)
 > FileHeader:               12 byte
 > Model Type Enum:          4 byte -> int
 > Vertices Stride:          4 byte -> long
 > Vertex Count:             4 byte -> long
 > Vertices Array:           4 byte -> float[Vertex Count * Vertices Stride]
 > Material Count:           4 byte -> long
 > Foreach Material Count:
 === > Indices Size:         4 byte -> long
 === > Indices Array:        4 byte -> long[Indices Size]

ARMATURE File Format (.armt)
 > FileHeader:                      12 byte
 > Bone Count:                      4 byte -> long
 > Foreach Bone Count:
 === > Bone Index:                  4 byte -> int
 === > Bone Name size:              4 byte -> long
 === > Bone Name:                   1 byte -> char[Bone Name Size]
 === > Bone Pose Transform:         4 byte -> float[16] (Matrix4f)
 === > Children Bone Indices Size:  4 byte -> long
 === > Children Bone Indices:       4 byte -> long[Children Bone Indices Size]

ANIMATION File Format (.anim)
 > FileHeader:                    12 byte
 > Frame Range:                   4 byte -> long[start,end]
 > Animation Frame Count:         4 byte -> long
 > BoneCurve Count(Per Frame):    4 byte -> long
 > Foreach Animation Frame Count:
 === > Frame Number:              4 byte -> long
 === > Foreach BoneCurve Count:
 === === > Bone Index:            4 byte -> long
 === === > Bone Location:         4 byte -> float[x,y,z]
 === === > Bone Scale:            4 byte -> float[x,y,z]
 === === > Bone Quaternion:       4 byte -> float[x,y,z,w]