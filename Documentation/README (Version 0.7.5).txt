OpenGL Exporter Version 0.7.5 (Additional Information)

Patch Notes:
 - Added to option to export tangents and bitangents
 - Changed almost all ModelType enums
 - Changed the ".model" file extension to ".mesh"
 - Added tangent and bitangent arrays to mesh file format

Data Structures:

Enum ModelType {
    xyz, xyzuv, xyzn, xyztbn, xyzuvn, xyzuvtbn, xyzrig, xyzuvrig, xyznrig, xyztbnrig, xyzuvnrig, xyzuvtbnrig
}

struct FileHeader { 
    int identifier; // 4 bytes
    int version[2]; // 8 bytes, Major and Minor version number
}

MESH File Format (.mesh) (Separate Vertex Buffers)
 > FileHeader:               12 byte
 > Model Type Enum:          4 byte -> int
 > Vertices Stride:          4 byte -> long
 > Vertex Count:             4 byte -> long
 > Vertices Array:           4 byte -> float[Vertex Count * 3]
 > UVs Array:                4 byte -> float[Vertex Count * 2] (Only if UVs are included)
 > Normals Array:            4 byte -> float[Vertex Count * 3] (Only if Normals are included)
 > Tangents Array:           4 byte -> float[Vertex Count * 3] (Only if Tangents are included)
 > Bitangents Array:         4 byte -> float[Vertex Count * 3] (Only if Tangents are included)
 > Weight Indices Array:     4 byte -> long [Vertex Count * 3] (Only if Rigging is included)
 > WeightsArray:             4 byte -> float[Vertex Count * 3] (Only if Rigging is included)
 > Material Count:           4 byte -> long
 > Foreach Material Count:
 === > Indices Size:         4 byte -> long
 === > Indices Array:        4 byte -> long[Indices Size]

MESH File Format (.mesh) (Single Vertex Buffer)
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
 > Time Range:                    4 byte -> float[start,end] // Frames or Seconds
 > Animation Frame Count:         4 byte -> long
 > BoneCurve Count(Per Frame):    4 byte -> long
 > Foreach Animation Frame Count:
 === > Frame Timestamp:           4 byte -> float // Frames or Seconds
 === > Foreach BoneCurve Count:
 === === > Bone Index:            4 byte -> long
 === === > Bone Location:         4 byte -> float[x,y,z]
 === === > Bone Scale:            4 byte -> float[x,y,z]
 === === > Bone Quaternion:       4 byte -> float[w,x,y,z]
