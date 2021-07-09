OpenGL Exporter Version 0.6 (Additional Information)

Patch Notes:
 - Added ANIMATION file format (.anim)
 - Added the "Y-axis as up" boolean in the settings (not fully supported yet)
 - Added "Armature" and "Animation" boolean in the settings to toggle exporting armatures and animations
 - Added "Frames" dropdown in the settings to choose which frames to export from the animation
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

ANIMATION File Format (.anim)
 > Exporter Version (Major):      4 byte -> int
 > Exporter Version (Minor):      4 byte -> int
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