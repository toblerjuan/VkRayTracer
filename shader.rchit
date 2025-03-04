#version 460
#extension GL_EXT_ray_tracing : enable

struct Material
{
		vec3 color;
		float reflectivity;
		float refractivity;
		float indexOfRefraction;
		float shininess;
};

struct RayPayload
{
		vec3 pos;
		vec3 normal;
		int anyHit;
		Material material;
};

struct Vertex
{
		vec3 pos;
		vec3 normal;
};

struct ReceivedVertex
{
		vec4 v;
		vec4 n;
};

layout (location = 0) rayPayloadInEXT RayPayload payload;
layout (set = 0, binding = 3) buffer Vertices { ReceivedVertex rv[]; } vertices;
layout (set = 0, binding = 4) buffer Indices { uint i[]; } indices;
layout (set = 0, binding = 5) buffer FirstVertices {uint fv[];} firstVertices;
layout (set = 0, binding = 6) buffer PrimOffsets {uint po[];} primOffsets;
layout (set = 0, binding = 7) buffer Materials { Material m[];} materials;

// Not sure how exactly this works.
hitAttributeEXT vec2 attribs;

Vertex unpack(uint idx, uint firstV)
{
		Vertex v;
		v.pos = vertices.rv[idx + firstV].v.xyz;
		v.normal = vertices.rv[idx + firstV].n.xyz;
		return v;
}

void main()
{
		// Choosing offset values
		uint firstV = firstVertices.fv[gl_InstanceCustomIndexEXT];
		uint primOff = primOffsets.po[gl_InstanceCustomIndexEXT];
		// Divide primitive offset by four since we receive it in bytes, but we
		// care only about the actual primitive number.
		primOff /= 4; 

		ivec3 ind = ivec3(indices.i[3 * gl_PrimitiveID + primOff], 
				indices.i[3 * gl_PrimitiveID + 1 + primOff],
				indices.i[3 * gl_PrimitiveID + 2 + primOff]);

		Vertex v0 = unpack(ind.x, firstV);
		Vertex v1 = unpack(ind.y, firstV);
		Vertex v2 = unpack(ind.z, firstV);

		// Computing and setting payload values
		const vec3 baryCoords = 
				vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
		payload.pos = gl_ObjectToWorldEXT  * vec4((baryCoords.x * v0.pos 
				+ baryCoords.y * v1.pos + baryCoords.z * v2.pos), 1.0);
		
		payload.normal = vec3((baryCoords.x * v0.normal + baryCoords.y * v1.normal
				+ baryCoords.z * v2.normal) * gl_WorldToObjectEXT);

		payload.anyHit = 1;

		payload.material = materials.m[gl_InstanceCustomIndexEXT];
}