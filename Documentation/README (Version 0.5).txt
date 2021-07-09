OpenGL Exporter Version 0.5 (Additional Information)

Patch Notes:
 - Added ARMATURE file format (.armt)
 - Added Model Type Enum to the MODEL file format
 - Added a "Selected + Children" option in the selection setting
 - Added the option to select which data to include in the vertex buffer
    - uv data
    - normal data
    - rigging data (bone weights and indices)
 - Many bug fixes

Data Structures:

Enum ModelType {
    xyz, xyzuv, xyztbn, xyzuvtbn, xyzRigged, xyzuvRigged, xyztbnRigged, xyzuvtbnRigged
}

MODEL File Format (.model)
 > Exporter Version (Major): 4 byte -> int
 > Exporter Version (Minor): 4 byte -> int
 > Model Type Enum:          4 byte -> int
 > Vertices Stride:          4 byte -> long
 > Vertices Size:            4 byte -> long
 > Vertices Array:           4 byte -> float[Vertices Size * Vertices Stride]
 > Material Count:           4 byte -> long
 > Foreach Material Count:
 === > Indices Size:         4 byte -> long
 === > Indices Array:        4 byte -> long[Indices Size]

ARMATURE File Format (.armt)
 > Exporter Version (Major):        4 byte -> int
 > Exporter Version (Minor):        4 byte -> int
 > Bone Count:                      4 byte -> long
 > Foreach Bone Count:
 === > Bone Index:                  4 byte -> int
 === > Bone Name size:              4 byte -> long
 === > Bone Name:                   1 byte -> char[Bone Name Size]
 === > Local Pose Transform:        4 byte -> float[16] (Matrix4f)
 === > Children Bone Indices Size:  4 byte -> long
 === > Children Bone Indices:       4 byte -> long[Children Bone Indices Size]