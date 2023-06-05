# Custom Exporter Blender Add-on v0.1.3

## Type Definitions

```
typedef enum VertexFlagBits : uint32 {
    VERTEX_FLAG_POSITION_BIT = 0x00000001,
    VERTEX_FLAG_COLOR_BIT = 0x00000002,
    VERTEX_FLAG_UV_BIT = 0x00000004,
    VERTEX_FLAG_NORMAL_BIT = 0x00000008,
    VERTEX_FLAG_TANGENT_BIT = 0x00000010,
    VERTEX_FLAG_BITANGENT_BIT = 0x00000020,
    VERTEX_FLAG_BONE_WEIGHT_BIT = 0x00000040,
    VERTEX_FLAG_BONE_WEIGHT_INDEX_BIT = 0x00000080,
} VertexFlagBits;
typedef uint32_t VertexFlag;
```

```
typedef enum VertexBufferFormat : uint32 { 
    VERTEX_BUFFER_FORMAT_INTERLEAVED = 0, 
    VERTEX_BUFFER_FORMAT_SEPARATED = 1,
} VertexBufferFormat;
```

```
typedef enum PrimitiveTopology : uint32 { 
    PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 0, 
    PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 1,
    PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_PRIMITIVE_RESTART = 2,
} PrimitiveTopology;
```

```
struct FileHeader {
    int32 identifier; // 4 byte -> Used to determine byte order
    int32 version[2]; // 8 byte -> Major and minor exporter version number
};
```

```
struct MeshHeader {
    VertexFlag vertexFlags;                // 4 byte
    VertexBufferFormat vertexBufferFormat; // 4 byte
    PrimitiveTopology primitiveTopology;   // 4 byte
    uint32 vertexStride;                   // 4 byte
    uint32 vertexCount;                    // 4 byte
    uint32 submeshCount;                   // 4 byte
};
```

## File Formats

```
MESH File Format (.mesh) // Interleaved Vertex Buffer Format
 > FileHeader:                      // 12 byte
 > MeshHeader:                      // 24 byte
 > Vertices Array:                  // 4 byte -> float32[MeshHeader.vertexCount * MeshHeader.vertexStride]
 > Foreach MeshHeader.submeshCount:
 == > IndexCount:                   // 4 byte -> uint32
 == > Indices Array:                // 4 byte -> uint32[IndexCount]
```

```
MESH File Format (.mesh) // Separated Vertex Buffer Format
 > FileHeader:                      // 12 byte
 > MeshHeader:                      // 24 byte
 > Positions Array:                 // 4 byte -> float32[MeshHeader.vertexCount * 3]
 > Colors Array:                    // 4 byte -> float32[MeshHeader.vertexCount * 3] (if VertexFlag includes VERTEX_FLAG_COLOR_BIT)
 > UVs Array:                       // 4 byte -> float32[MeshHeader.vertexCount * 2] (if VertexFlag includes VERTEX_FLAG_UV_BIT)
 > Normals Array:                   // 4 byte -> float32[MeshHeader.vertexCount * 3] (if VertexFlag includes VERTEX_FLAG_NORMAL_BIT)
 > Tangents Array:                  // 4 byte -> float32[MeshHeader.vertexCount * 3] (if VertexFlag includes VERTEX_FLAG_TANGENT_BIT)
 > Bitangents Array:                // 4 byte -> float32[MeshHeader.vertexCount * 3] (if VertexFlag includes VERTEX_FLAG_BITANGENT_BIT)
 > Foreach MeshHeader.submeshCount:
 == > IndexCount:                   // 4 byte -> uint32
 == > Indices Array:                // 4 byte -> uint32[IndexCount]
```
