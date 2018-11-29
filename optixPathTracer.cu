/* 
 * Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include "main.h"
#include "random.h"


using namespace optix;

struct PerRayData_pathtrace
{
    float3 result;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    unsigned int seed;
    int depth;
    int countEmitted;
    int done;
};

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

static __device__ __inline__ float2 concentric_sample_disk(const float2& uOffset)
{
    // ConcentricSampleDisk
    // a) Map uniform random number to [-1,1]^2
    //float2 uOffset = 2.f * u - make_float2(1.f, 1.f);

    // b) Handle degeneracy at origin
    if (uOffset.x == 0 && uOffset.y == 0)
        return make_float2(0.f, 0.f);

    // c) Apply concentric mapping to point
    float theta;
    float r;
    if(abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = M_PI_4f * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = M_PI_2f - M_PI_4f * (uOffset.x / uOffset.y);
    }


    return r * make_float2(cos(theta), sin(theta));
}

static __device__ __inline__ float2 get_disk_sample(const float2& u,
    const float maxwidth,
    const float maxheight,
    const uchar2& index)
{

    float w = (-(1 / 2) + ((index.x + 0.5) / maxwidth));
        
    float h = ((1 / 2) - ((index.y + 0.5) / maxheight));

    // a) Map uniform random number to [-1,1]^2
    float2 uOffset = 2.f * u - make_float2(1.f, 1.f);

    float2 image_point = make_float2(w, h);

    image_point += uOffset;

    return concentric_sample_disk(image_point);
}


// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );




//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // Russian roulette termination 
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
        seed = prd.seed;
    } while (--samples_per_pixel);

    //
    // Update the output buffer
    //
    float3 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);

    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
}

rtDeclareVariable(float, f_length, , );
rtDeclareVariable(float, lens_rad, , );

RT_PROGRAM void thin_lens_camera()
{


    // If lens_radius is 0, treat as pinhole
    if(lens_rad < .00001f)
    {
        pathtrace_camera();
        return;
    }

    // Get ray direction of eye to image plane in the same way as before
    size_t2 screen = output_buffer.size();
    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 d = (make_float2(launch_index)) * inv_screen - 1.f;

    float3 init_ray_origin = eye;
    float3 init_ray_direction = normalize(d.x*U + d.y*V + W);

    float ft = f_length / dot(init_ray_direction,normalize(W));
    float3 pFocus = init_ray_origin + init_ray_direction * ft;

    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
    float3 result = make_float3(0.0f);

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;

        float2 sample_point = make_float2(rnd(seed), rnd(seed));

        float2 disc_point = get_disk_sample(sample_point,
            sqrt_num_samples,
            sqrt_num_samples,
            make_uchar2(x,y));

        float2 p_lens = lens_rad * disc_point;

        // 2) Compute point on plane of focus
        float3 ray_origin = init_ray_origin + p_lens.x * U + p_lens.y * V;
        float3 ray_direction = normalize(pFocus - ray_origin);
        // printf("pre o %f %f %f\n", init_ray_origin.x, init_ray_origin.y, init_ray_origin.z);
        // printf("post o %f %f %f\n", ray_origin.x, ray_origin.y, ray_origin.z);
        // printf("pre d %f %f %f\n", init_ray_direction.x, init_ray_direction.y, init_ray_direction.z);
        // printf("post d %f %f %f\n", ray_direction.x, ray_direction.y, ray_direction.z);


        // float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        // float2 d = pixel + jitter*jitter_scale;
        // float3 ray_origin = eye;
        // float3 ray_direction = normalize(-d.x*U - d.y*V + W);

        // Initialze per-ray data
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        for(;;)
        {
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);

            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // Russian roulette termination 
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += prd.result;
        seed = prd.seed;
    } while (--samples_per_pixel);

    //
    // Update the output buffer
    //
    float3 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);

    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        output_buffer[launch_index] = make_float4( lerp( old_color, pixel_color, a ), 1.0f );
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
    }
}



//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );

RT_PROGRAM void diffuseEmitter()
{
    current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
    current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );


RT_PROGRAM void diffuse()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float z1=rnd(current_prd.seed);
    float z2=rnd(current_prd.seed);
    float3 p;
    cosine_sample_hemisphere(z1, z2, p);
    optix::Onb onb( ffnormal );
    onb.inverse_transform( p );
    current_prd.direction = p;

    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation * diffuse_color;
    current_prd.countEmitted = false;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

        // cast shadow ray
        if ( nDl > 0.0f && LnDl > 0.0f )
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon );
            rtTrace(top_object, shadow_ray, shadow_prd);

            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;
}


RT_PROGRAM void mirror()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;
    current_prd.direction = reflect(ray.direction, ffnormal );


    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation * diffuse_color;
    current_prd.countEmitted = true;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

    }

    current_prd.radiance = result;

}

/*
Glass instructions:
If there is total internal reflection
    assign the reflection of wo to *wi
    set the *pdf to 1
    return reflectance / abs_cos_theta(*wi)
Else
    Compute Schlick's reflection coefficient RR
    If coin_flip(R)
        assign the reflection of wo to *wi
        set the *pdf to R
        return R * reflectance / abs_cos_theta(*wi)
    Else
        assign the refraction of wo to *wi
        set the *pdf to 1-R
        return (1-R) * transmittance / abs_cos_theta(*wi) / eta^2 (where eta^2 is the same as in the refraction function)
*/

rtDeclareVariable(float,        refraction_index, , );

RT_PROGRAM void glass()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    //EDIT HERE TO IMPLEMENT GLASS

    float3 i = ray.direction;                                            // incident direction
    float3 t;                                                            // transmission direction
    float3 r;                                                            // reflection direction
    float3 n = world_shading_normal;

    if ( refract(t, i, n, refraction_index) )
    {
        current_prd.direction = t;
    } 
    else
    {
        current_prd.direction = reflect(ray.direction, ffnormal );
    }


    // NOTE: For diffuse, f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    current_prd.attenuation = current_prd.attenuation * diffuse_color;
    current_prd.countEmitted = true;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

        // cast shadow ray
        if ( nDl > 0.0f && LnDl > 0.0f )
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon );
            rtTrace(top_object, shadow_ray, shadow_prd);

            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                result += light.emission * weight;
            }
        }
    }

    current_prd.radiance = result;


}


rtBuffer<float4> planes;
RT_PROGRAM void chull_intersect(int primIdx)
{
    int n = planes.size();
    float t0 = -exp(800.0); //FLT_MIN
    float t1 = exp(800.0); //FLT_MAX
    float3 t0_normal = make_float3(0);
    float3 t1_normal = make_float3(0);
    for(int i = 0; i < n && t0 < t1; ++i ) {
    float4 plane = planes[i];
    float3 n = make_float3(plane);
    float  d = plane.w;

    float denom = dot(n, ray.direction);
    float t = -(d + dot(n, ray.origin))/denom;
    if( denom < 0){
        // enter
        if(t > t0){
        t0 = t;
        t0_normal = n;
        }
    } else {
        //exit
        if(t < t1){
        t1 = t;
        t1_normal = n;
        }
    }
    }
    if(t0 > t1)
    return;

// intersection program continued from above
    if(rtPotentialIntersection( t0 )){
    shading_normal = geometric_normal = t0_normal;
    rtReportIntersection(0);
    } else if(rtPotentialIntersection( t1 )){
    shading_normal = geometric_normal = t1_normal;
    rtReportIntersection(0);
    }
}




RT_PROGRAM void chull_bounds (int primIdx, float result[6])
{
   optiX::Aabb* aabb = (Aabb*) result;
//    aabb->m_min = chull_bbmin;
//    aabb->m_max = chull_bbmax;
}




//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    current_prd.radiance = bg_color;
    current_prd.done = true;
}


rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  current_prd.radiance = make_float3( tex2D(envmap, u, v) );
  current_prd.done = true;

}
  

RT_PROGRAM void tex()
{

    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;
    current_prd.direction = ffnormal ;


  float theta = atan2f( current_prd.direction.x, current_prd.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( current_prd.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  current_prd.radiance = make_float3( tex2D(envmap, u, v) );
  current_prd.done = true;

}
