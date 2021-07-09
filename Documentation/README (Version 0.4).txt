OpenGL Exporter Version 0.4 (Additional Information)

Patch Notes:
 - Changed MODEL file extension to (.model).
 - Minor bug fixes

Data Structure:

MODEL File Format (.model)
 > Exporter Version (Major): 4 byte -> int
 > Exporter Version (Minor): 4 byte -> int
 > Vertices Stride:          4 byte -> long
 > Vertices Size:            4 byte -> long
 > Vertices Array:           4 byte -> float[Vertices Size * Vertices Stride]
 > Material Count:           4 byte -> long
 > Foreach Material Count:
 === > Indices Size:         4 byte -> long
 === > Indices Array:        4 byte -> long[Indices Size]