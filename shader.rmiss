#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : require

struct Material {
    vec3 color;
    float reflectivity;
    float refractivity;
    float IOR;
    float shininess;
};

struct RayPayload {
    vec3 pos;
    vec3 normal;
    int anyhit;
    Material material;
};

layout (location = 0) rayPayloadInEXT RayPayload rayPayload;

void main()
{
	rayPayload.anyhit = 0;
}