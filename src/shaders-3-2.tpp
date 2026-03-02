static const char *cube_vertex_3_2 =
R"(#version 320 es
in vec3 position;
in vec2 uvPosition;

out vec2 uvpos;
out vec3 vPos;

void main() {
    vPos = position;
    uvpos = uvPosition;
})";

static const char *cube_tcs_3_2 =
R"(#version 320 es
layout(vertices = 3) out;

in vec2 uvpos[];
in vec3 vPos[];

out vec3 tcPosition[];
out vec2 uv[];

#define ID gl_InvocationID

uniform int deform;
uniform int light;

void main() {
    tcPosition[ID] = vPos[ID];
    uv[ID] = uvpos[ID];

    if(ID == 0){
        /* deformation requires tessellation
           and lighting even higher degree to
           make lighting smoother */

        float tessLevel = 1.0f;
        if(deform > 0)
            tessLevel = 30.0f;
        if(light > 0)
            tessLevel = 50.0f;

        gl_TessLevelInner[0] = tessLevel;
        gl_TessLevelOuter[0] = tessLevel;
        gl_TessLevelOuter[1] = tessLevel;
        gl_TessLevelOuter[2] = tessLevel;
    }
})";

static const char *cube_tes_3_2 =
R"(#version 320 es
layout(triangles) in;

in vec3 tcPosition[];
in vec2 uv[];

out vec2 tesuv;
out vec3 tePosition;

uniform mat4 model;
uniform mat4 VP;
uniform int  deform;
uniform float ease;

vec2 interpolate2D(vec2 v0, vec2 v1, vec2 v2) {
    return vec2(gl_TessCoord.x) * v0
         + vec2(gl_TessCoord.y) * v1
         + vec2(gl_TessCoord.z) * v2;
}

vec3 interpolate3D(vec3 v0, vec3 v1, vec3 v2) {
    return vec3(gl_TessCoord.x) * v0
         + vec3(gl_TessCoord.y) * v1
         + vec3(gl_TessCoord.z) * v2;
}


vec3 tp;
void main() {
    tesuv = interpolate2D(uv[0], uv[1], uv[2]);

    tp = interpolate3D(tcPosition[0], tcPosition[1], tcPosition[2]);
    tp = (model * vec4(tp, 1.0)).xyz;

    if(deform > 0) {
        float r = 0.5;
        float d = distance(tp.xz, vec2(0, 0));
        float scale = 1.0;
        if(deform == 1)
            scale = r / d;
        else
            scale = d / r;

        scale = pow(scale, ease);
        tp = vec3(tp[0] * scale, tp[1], tp[2] * scale);
    }

    tePosition = tp;
    gl_Position = VP * vec4 (tp, 1.0);
})";

static const char* cube_geometry_3_2 =
R"(#version 320 es
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec2 tesuv[3];
in vec3 tePosition[3];
uniform int  light;
uniform float cameraYOffset;
uniform float cubeVerticalOffset;  // Y position of the cube being rendered
out vec2 guv;
out vec3 colorFactor;
#define AL 0.3    // ambient lighting
#define DL (1.0-AL) // diffuse lighting
void main() {
    // Light position at the same Y as the cube being rendered
    vec3 lightSource = vec3(0, cubeVerticalOffset, 2);
    vec3 lightNormal = normalize(vec3(0, 0, 1));
    
    if(light == 1) {
        vec3 A = tePosition[2] - tePosition[0];
        vec3 B = tePosition[1] - tePosition[0];
        vec3 N = normalize(cross(A, B));
        vec3 center = (tePosition[0] + tePosition[1] + tePosition[2]) / 3.0;
        float d = distance(center, lightSource);
        float ambient_coeff = pow(clamp(2.0 / d, 0.0, 1.0), 10.0);
        float value = clamp(pow(abs(dot(N, lightNormal)), 1.5), 0.0, 1.0);
        float df = AL * ambient_coeff + DL * value;
        colorFactor = vec3(df, df, df);
    }
    else
        colorFactor = vec3(1.0, 1.0, 1.0);
    gl_Position = gl_in[0].gl_Position;
    guv = tesuv[0];
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    guv = tesuv[1];
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    guv = tesuv[2];
    EmitVertex();
})";


static const char *cube_vertex_3_2_simple = R"(
#version 320 es
layout(location = 0) in highp vec2 position;
layout(location = 1) in highp vec2 uvPosition;

uniform highp mat4 VP;
uniform highp mat4 model;

out highp vec2 guv;
out highp vec3 colorFactor;

void main() {
    guv = uvPosition;
    colorFactor = vec3(1.0, 1.0, 1.0);
    gl_Position = VP * model * vec4(position, 0.0, 1.0);
}
)";


static const char *cube_fragment_3_2_orginal =
R"(#version 320 es
precision highp float;
precision highp sampler2D;
in highp vec2 guv;
in highp vec3 colorFactor;
layout(location = 0) out highp vec4 outColor;
uniform highp sampler2D smp;
void main() {
    highp vec4 texColor = texture(smp, guv);
    outColor = vec4(texColor.rgb * colorFactor, texColor.a);
})";    


static const char *cube_fragment_3_2 =
R"(#version 320 es
precision highp float;
precision highp sampler2D;
in highp vec2 guv;
in highp vec3 colorFactor;
layout(location = 0) out highp vec4 outColor;
uniform highp sampler2D smp;
uniform highp float u_time;
uniform highp float u_brightness;
// Function to calculate rounded corner mask
float roundedCornerMask(vec2 uv, float radius) {
    vec2 centered_uv = uv - 0.5;
    vec2 corner_dist = abs(centered_uv) - (0.5 - radius);
    if (corner_dist.x <= 0.0 && corner_dist.y <= 0.0) {
        return 1.0;
    }
    if (corner_dist.x > 0.0 && corner_dist.y > 0.0) {
        float dist_to_corner = length(corner_dist);
        return smoothstep(radius + 0.01, radius - 0.01, dist_to_corner);
    }
    return 1.0;
}
// Function to calculate edge anti-aliasing mask
float edgeAntiAliasMask(vec2 uv) {
    float min_edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float aa_width = 0.01;
    return smoothstep(0.0, aa_width, min_edge_dist);
}
// Function to calculate edge glow
float edgeGlow(vec2 uv, float radius) {
    // Distance from edges (sides)
    float edge_dist_x = min(uv.x, 1.0 - uv.x);
    float edge_dist_y = min(uv.y, 1.0 - uv.y);
    float edge_dist = min(edge_dist_x, edge_dist_y);
   
    // Distance from corners - calculate actual distance to rounded corner edge
    vec2 centered_uv = uv - 0.5;
    vec2 corner_dist = abs(centered_uv) - (0.5 - radius);
   
    // Check if we're in corner region
    bool in_corner = corner_dist.x > 0.0 && corner_dist.y > 0.0;
   
    float distance_to_use;
    if (in_corner) {
        // In corner: distance from the rounded arc
        float dist_to_corner_center = length(corner_dist);
        distance_to_use = abs(dist_to_corner_center - radius);
    } else {
        // On straight edge
        distance_to_use = edge_dist;
    }
   
    // Create glow falloff
    float glow_width = 0.03; // Width of the glow effect
    float glow = smoothstep(glow_width, 0.0, distance_to_use);
   
    return glow;
}
void main() {
    // Ripple distortion in the middle
    vec2 center = vec2(0.5);
    vec2 dir = guv - center;
    float dist = length(dir);
    float ripple = sin(dist * 20.0 - u_time * 5.0) * 0.01;
    vec2 distorted_uv = guv + normalize(dir) * ripple;
   
    highp vec4 texColor = texture(smp, distorted_uv);
   
    // Apply rounded corners
    float corner_radius = 0.08;
    float corner_mask = roundedCornerMask(guv, corner_radius);
    float edge_mask = edgeAntiAliasMask(guv);
    float final_mask = min(corner_mask, edge_mask);
   
    // Calculate edge glow
    float glow = edgeGlow(guv, corner_radius);
   
    // Glow color (can be customized)
    vec3 glow_color = vec3(0.4, 0.6, 1.0); // Cyan/blue glow
    float glow_intensity = 0.6; // How bright the glow is
   
    // Optional: Animate glow with pulse
    float pulse = sin(u_time * 2.0) * 0.2 + 0.8; // Oscillates between 0.6 and 1.0
    glow *= pulse;
   
    // Apply lighting and glow
    vec3 finalColor = texColor.rgb * colorFactor;
    finalColor += glow_color * glow * glow_intensity;
    finalColor *= u_brightness;
   
    float finalAlpha = texColor.a * final_mask;
   
    outColor = vec4(finalColor, finalAlpha);
})";


static const char *cube_fragment_3_2_tron =
R"(#version 320 es
precision highp float;
precision highp sampler2D;
in highp vec2 guv;
in highp vec3 colorFactor;
layout(location = 0) out highp vec4 outColor;
uniform highp sampler2D smp;
uniform highp float u_time;
uniform highp float u_brightness;

// TRON-STYLE EDGE CONFIGURATION
#define LINE_WIDTH 0.008                      // Thickness of traveling line
#define LINE_LENGTH 0.12                      // Length of traveling segment
#define GLOW_WIDTH 0.55                      // Glow spread
#define ANIMATION_SPEED 0.2                   // Speed of travel
#define BASE_EDGE_GLOW 0.7                    // Subtle constant edge glow
#define NUM_LINES 12                           // Number of traveling lines

// LINE COLORS - Blue, Red, Green, White
#define LINE_COLOR_0 vec3(0.0, 0.5, 1.0)     // Blue
#define LINE_COLOR_1 vec3(0.0, 0.5, 1.0)      // Red
#define LINE_COLOR_2 vec3(0.0, 0.5, 1.0)      // Green
#define LINE_COLOR_3 vec3(0.0, 0.5, 1.0)      // White

float roundedCornerMask(vec2 uv, float radius) {
    vec2 centered_uv = uv - 0.5;
    vec2 corner_dist = abs(centered_uv) - (0.5 - radius);
    if (corner_dist.x <= 0.0 && corner_dist.y <= 0.0) {
        return 1.0;
    }
    if (corner_dist.x > 0.0 && corner_dist.y > 0.0) {
        float dist_to_corner = length(corner_dist);
        return smoothstep(radius + 0.01, radius - 0.01, dist_to_corner);
    }
    return 1.0;
}

float edgeAntiAliasMask(vec2 uv) {
    float min_edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float aa_width = 0.01;
    return smoothstep(0.0, aa_width, min_edge_dist);
}

float distanceToEdge(vec2 uv) {
    float dist_left = uv.x;
    float dist_right = 1.0 - uv.x;
    float dist_bottom = uv.y;
    float dist_top = 1.0 - uv.y;
    return min(min(dist_left, dist_right), min(dist_bottom, dist_top));
}

float uvToPerimeter(vec2 uv) {
    vec2 p = uv;
    float edge_dist = distanceToEdge(uv);
    
    if (edge_dist == p.y) {
        return p.x;  // Bottom edge
    } else if (edge_dist == (1.0 - p.x)) {
        return 1.0 + p.y;  // Right edge
    } else if (edge_dist == (1.0 - p.y)) {
        return 3.0 - p.x;  // Top edge
    } else {
        return 4.0 - p.y;  // Left edge
    }
}

// Get color for each line based on index
vec3 getLineColor(int index) {
    if (index == 0) return LINE_COLOR_0;
    if (index == 1) return LINE_COLOR_1;
    if (index == 2) return LINE_COLOR_2;
    return LINE_COLOR_3;
}

void main() {
    vec2 center = vec2(0.5);
    vec2 dir = guv - center;
    float dist = length(dir);
    float ripple = sin(dist * 20.0 - u_time * 5.0) * 0.01;
    vec2 distorted_uv = guv + normalize(dir) * ripple;
    
    highp vec4 texColor = texture(smp, distorted_uv);
    
    float corner_radius = 0.08;
    float corner_mask = roundedCornerMask(guv, corner_radius);
    float edge_mask = edgeAntiAliasMask(guv);
    float final_mask = min(corner_mask, edge_mask);
    
    vec3 finalColor = texColor.rgb * colorFactor;
    
    // TRON TRAVELING LINES (4 DIFFERENT COLORED LINES)
    float edge_dist = distanceToEdge(guv);
    
    if (edge_dist < 0.05) {
        float perimeter_pos = uvToPerimeter(guv);
        float spacing = 4.0 / float(NUM_LINES);
        
        float edge_falloff = smoothstep(0.05, 0.0, edge_dist);
        float sharp_line = smoothstep(LINE_WIDTH * 2.0, 0.0, edge_dist);
        float soft_glow = smoothstep(GLOW_WIDTH, 0.0, edge_dist);
        
        // Create 4 lines, each with its own color
        for (int i = 0; i < NUM_LINES; i++) {
            float line_pos = mod(u_time * ANIMATION_SPEED + float(i) * spacing, 4.0);
            float dist_to_line = abs(perimeter_pos - line_pos);
            dist_to_line = min(dist_to_line, abs(dist_to_line - 4.0));
            
            float line_intensity = smoothstep(LINE_LENGTH, 0.0, dist_to_line);
            float trail_intensity = smoothstep(LINE_LENGTH * 2.0, LINE_LENGTH, dist_to_line) * 0.3;
            line_intensity = max(line_intensity, trail_intensity);
            
            // Get the color for this specific line
            vec3 this_line_color = getLineColor(i);
            vec3 this_glow_color = this_line_color * 0.6;
            
            // Apply this line's color
            finalColor += this_line_color * sharp_line * line_intensity * edge_falloff * 2.0;
            finalColor += this_glow_color * soft_glow * line_intensity * edge_falloff;
        }
        
        // Add subtle constant edge glow (white)
        finalColor += vec3(0.5, 0.5, 0.5) * edge_falloff * BASE_EDGE_GLOW * 0.3;
    }
     finalColor *= u_brightness;
    float finalAlpha = texColor.a * final_mask;
    outColor = vec4(finalColor, finalAlpha);
})";


// Vertex shader - updated to pass world position
static const char *cube_cap_vertex = R"(
#version 100
attribute mediump vec2 position;
attribute mediump vec2 uvPosition;

uniform mat4 VP;
uniform mat4 model;

varying mediump vec2 uvpos;
varying mediump vec3 worldPos;

void main() {
    uvpos = uvPosition;
    vec4 worldPosition = model * vec4(position.x, 0.0, position.y, 1.0);
    worldPos = worldPosition.xyz;
    gl_Position = VP * worldPosition;
}
)";

static const char *cube_cap_fragment = R"(
#version 100
precision mediump float;
varying mediump vec2 uvpos;
varying mediump vec3 worldPos;
uniform sampler2D smp;
uniform float cap_alpha;
uniform float time;
uniform float u_num_sides;

// Simple rounded corner mask for square UV space
float roundedCornerMask(vec2 uv, float radius) {
    vec2 centered_uv = uv - 0.5;
    vec2 corner_dist = abs(centered_uv) - (0.5 - radius);
    
    if (corner_dist.x <= 0.0 && corner_dist.y <= 0.0) {
        return 1.0;
    }
    
    if (corner_dist.x > 0.0 && corner_dist.y > 0.0) {
        float dist_to_corner = length(corner_dist);
        return smoothstep(radius + 0.01, radius - 0.01, dist_to_corner);
    }
    
    return 1.0;
}

// Edge anti-aliasing
float edgeAntiAliasMask(vec2 uv) {
    float min_edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float aa_width = 0.01;
    return smoothstep(0.0, aa_width, min_edge_dist);
}

void main() {
    // Calculate distance from center using UV coordinates
    vec2 centerUV = uvpos - vec2(0.5, 0.5);
    float dist = length(centerUV) * 2.0;
    
    // Wave parameters
    float frequency = 40.0;
    float speed = 1.8;
    float amplitude = 0.1;
    
    // Calculate wave height
    float height = sin(dist * frequency - time * speed) * amplitude;
    
    // Calculate gradients for normal mapping
    float delta = 0.01;
    
    // X gradient
    vec2 uvX1 = uvpos + vec2(delta, 0.0);
    vec2 uvX2 = uvpos - vec2(delta, 0.0);
    float distX1 = length((uvX1 - vec2(0.5, 0.5)) * 2.0);
    float distX2 = length((uvX2 - vec2(0.5, 0.5)) * 2.0);
    float hX1 = sin(distX1 * frequency - time * speed) * amplitude;
    float hX2 = sin(distX2 * frequency - time * speed) * amplitude;
    float dx = (hX1 - hX2) / (2.0 * delta);
    
    // Y gradient
    vec2 uvY1 = uvpos + vec2(0.0, delta);
    vec2 uvY2 = uvpos - vec2(0.0, delta);
    float distY1 = length((uvY1 - vec2(0.5, 0.5)) * 2.0);
    float distY2 = length((uvY2 - vec2(0.5, 0.5)) * 2.0);
    float hY1 = sin(distY1 * frequency - time * speed) * amplitude;
    float hY2 = sin(distY2 * frequency - time * speed) * amplitude;
    float dy = (hY1 - hY2) / (2.0 * delta);
    
    // Calculate normal from gradients
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    
    // Animated light direction
    vec3 lightDir = normalize(vec3(0.3, sin(time * 0.2), 0.5));
    
    // Calculate lighting
    float brightness = clamp(exp(dot(normal, lightDir)) * 0.5, 0.0, 1.0);
    
    // Get base color from texture
    vec4 texColor = texture2D(smp, uvpos);
    
    // Apply lighting
    vec3 finalColor = texColor.rgb * brightness;
    
    // Apply simple rounded corners (works in UV space)
    float corner_radius = 0.6;
    float corner_mask = roundedCornerMask(uvpos, corner_radius);
    float edge_mask = edgeAntiAliasMask(uvpos);
    float final_mask = min(corner_mask, edge_mask);
    
    // Combine texture alpha with corner mask
    float finalAlpha = texColor.a * final_mask;
    
    gl_FragColor = vec4(finalColor, finalAlpha);
}
)";

// Background Vertex Shader - Simple fullscreen quad
static const char *background_vertex_shader = R"(
#version 100
attribute vec2 position;
varying vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_uv = position * 0.5 + 0.5;
}
)";




static const char *background_fragment_shader = R"(
#version 100
precision mediump float;
uniform float u_time;
uniform vec2 u_resolution;
varying vec2 v_uv;

// Star Nest by Pablo Roman Andrioli
// License: MIT
#define iterations 10
#define formuparam 0.28
#define volsteps 20
#define stepsize 0.1
#define zoom 0.800
#define tile 0.850
#define speed 0.01
#define brightness 0.0005
#define darkmatter 0.900
#define distfading 0.630
#define saturation 1.90

// TRON-STYLE EDGE CONFIGURATION
#define LINE_WIDTH 0.008
#define LINE_LENGTH 0.12
#define GLOW_WIDTH 0.55
#define ANIMATION_SPEED 0.2
#define BASE_EDGE_GLOW 0.5
#define NUM_LINES 12
#define EDGE_MARGIN 0.0

// LINE COLOR - Blue
#define LINE_COLOR vec3(0.0, 0.5, 1.0)

// NEW TRON ELEMENTS
#define GRID_SIZE 0.05
#define GRID_WIDTH 0.001
#define GRID_GLOW 0.15
#define SCANLINE_SPEED 0.3
#define HEX_GLOW 0.2
#define PI 3.14159265359

float distanceToEdge(vec2 uv, vec2 resolution) {
    vec2 pixel_pos = uv * resolution;
    float dist_left = pixel_pos.x - EDGE_MARGIN;
    float dist_right = resolution.x - EDGE_MARGIN - pixel_pos.x;
    float dist_bottom = pixel_pos.y - EDGE_MARGIN;
    float dist_top = resolution.y - EDGE_MARGIN - pixel_pos.y;
    return min(min(dist_left, dist_right), min(dist_bottom, dist_top)) / resolution.x;
}



float uvToPerimeter(vec2 uv, vec2 resolution) {
    vec2 pixel_pos = uv * resolution;
    vec2 margin = vec2(EDGE_MARGIN);
    vec2 inner_size = resolution - 2.0 * margin;
    vec2 p = (pixel_pos - margin) / inner_size.x;
    
    float dist_left = p.x;
    float dist_right = (inner_size.x / inner_size.x) - p.x;
    float dist_bottom = p.y * (inner_size.y / inner_size.x);
    float dist_top = (inner_size.y / inner_size.x) - p.y * (inner_size.y / inner_size.x);
    
    float edge_dist = min(min(dist_left, dist_right), min(dist_bottom, dist_top));
    float perimeter_length = 2.0 * (1.0 + inner_size.y / inner_size.x);
    
    if (edge_dist == dist_bottom) {
        return p.x / perimeter_length;
    } else if (edge_dist == dist_right) {
        return (1.0 + p.y * (inner_size.y / inner_size.x)) / perimeter_length;
    } else if (edge_dist == dist_top) {
        return (1.0 + (inner_size.y / inner_size.x) + (1.0 - p.x)) / perimeter_length;
    } else {
        return (2.0 + (inner_size.y / inner_size.x) + ((inner_size.y / inner_size.x) - p.y * (inner_size.y / inner_size.x))) / perimeter_length;
    }
}

// Rotation
vec2 rotate2D(vec2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Grid pattern
float grid(vec2 uv) {
    vec2 grid_uv = fract(uv / GRID_SIZE);
    vec2 grid_line = smoothstep(GRID_WIDTH, 0.0, abs(grid_uv - 0.5) - 0.5 + GRID_WIDTH);
    return max(grid_line.x, grid_line.y);
}

// Hexagon shape
float hexagon(vec2 p, float r) {
    vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

// Spinning hexagon with glow
vec3 spinningHexagon(vec2 uv, vec2 center, float size, float atime, float aspeed) {
    vec2 p = uv - center;
    p = rotate2D(p, atime * aspeed);
    
    float d = hexagon(p, size);
    float outline = smoothstep(0.003, 0.0, abs(d));
    float glow = smoothstep(0.04, 0.0, abs(d)) * 0.3;
    
    return LINE_COLOR * (outline + glow);
}
// Pinwheel/Turbine shape - matches the uploaded logo with dual-lip blades
// Pinwheel/Turbine shape - matches the uploaded logo with dual-lip blades
vec3 pinwheel(vec2 uv, vec2 center, float size, float atime, float aspeed) {
    vec2 p = uv - center;
    p = rotate2D(p, atime * aspeed);
    
    float angle = atan(p.y, p.x);
    float dist = length(p);
    
    vec3 color = vec3(0.0);
    
    // Define colors for each blade
    vec3 red = vec3(0.9, 0.15, 0.1);
    vec3 yellow = vec3(1.0, 0.95, 0.2);
    vec3 orange = vec3(1.0, 0.45, 0.0);
    
    // Create 6 curved blades with alternating colors
    for (int i = 0; i < 6; i++) {
        float blade_angle = float(i) * PI / 3.0;
        float angle_diff = angle - blade_angle;
        
        // Normalize angle difference to -PI to PI
        angle_diff = mod(angle_diff + PI, 2.0 * PI) - PI;
        
        // Create smooth S-curve
        float normalized_dist = dist / size;
        
        // Primary curve
        float curve = -1.2 * sin(normalized_dist * 2.8 + 0.5) * normalized_dist;
        
        // Base blade width that WIDENS toward the tip
        float width_curve = sin(normalized_dist * PI);
        
        // Blade gets wider as it extends outward (from 0.6 to tip)
        float widening = 1.0;
        if (normalized_dist > 0.6) {
            widening = 1.0 + ((normalized_dist - 0.6) / 0.4) * 0.8; // Gets up to 1.8x wider
        }
        
        // Add edge dips - creates narrowing at start
        float edge_dip_start = smoothstep(0.0, 0.1, normalized_dist);
        
        // Create the center notch at the outer edge (creating two lips)
       // Create the dual-lip notch at the outer edge
        float notch = 0.0;
        if (normalized_dist > 0.75) {
            // Position along the outer portion of the blade
            float outer_progress = (normalized_dist - 0.75) / 0.25;
            
            // Distance from center line of blade
            float lateral_dist = abs(angle_diff - curve);
            
            // Create two "lips" - one on each side of center
            // The notch is deepest slightly off-center on each side
            float lip_left = smoothstep(0.01, 0.12, lateral_dist) * smoothstep(0.20, 0.15, lateral_dist);
            float lip_right = smoothstep(0.01, 0.12, lateral_dist) * smoothstep(0.20, 0.15, lateral_dist);
            
            // Combine lips to create the dual-notch effect
            // Stronger near the tip
            notch = (lip_left + lip_right) * outer_progress * 0.35;
        }
        
        // Combine base width with widening, edge dips, and center notch
        float blade_width = (0.25 + width_curve * 0.25) * widening * (0.5 + edge_dip_start * 0.5) * (1.0 - notch);
        
        float blade_dist = abs(angle_diff - curve);
        
        // Sharper blade edges
        float blade_mask = 1.0 - smoothstep(blade_width * 0.5, blade_width, blade_dist);
        
        // Better radial falloff
        float inner_radius = 0.05;
        float outer_radius = 1.05;
        
        float outer_fade = smoothstep(outer_radius, outer_radius - 0.15, normalized_dist);
        float inner_fade = 1.0 - smoothstep(inner_radius + 0.08, inner_radius, normalized_dist);
        
        float radial_mask = outer_fade * inner_fade;
        
        // Assign colors: red, orange, yellow pattern
        vec3 blade_color;
        int color_index = int(mod(float(i), 3.0));
        if (color_index == 0) {
            blade_color = red;
        } else if (color_index == 1) {
            blade_color = orange;
        } else {
            blade_color = yellow;
        }
        
        // Apply blade color with intensity boost
        color += blade_color * blade_mask * radial_mask * 1.1;
    }
    
    // Add blue hexagonal outer border
    float hex_d = hexagon(p, size * 1.15);
    float hex_outline = smoothstep(0.005, 0.0, abs(hex_d));
    float hex_glow = smoothstep(0.01, 0.0, abs(hex_d)) * 0.3;
    color += LINE_COLOR * (hex_outline * 2.0 + hex_glow);
    
    return color;
}
// Triangle shape
float triangle(vec2 p, float r) {
    float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) {
        p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    }
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

// Rotating triangle
vec3 rotatingTriangle(vec2 uv, vec2 center, float size, float atime, float aspeed) {
    vec2 p = uv - center;
    p = rotate2D(p, atime * aspeed);
    
    float d = triangle(p, size);
    float outline = smoothstep(0.003, 0.0, abs(d));
    float glow = smoothstep(0.04, 0.0, abs(d)) * 0.3;
    
    return LINE_COLOR * (outline + glow);
}

// Pulsing hexagon
vec3 pulsingHexagon(vec2 uv, vec2 center, float baseSize, float atime, float aspeed) {
    vec2 p = uv - center;
    float pulse = sin(atime * aspeed) * 0.5 + 0.5;
    float size = baseSize * (0.7 + pulse * 0.3);
    
    float d = hexagon(p, size);
    float outline = smoothstep(0.003, 0.0, abs(d));
    float glow = smoothstep(0.04, 0.0, abs(d)) * 0.3 * pulse;
    
    return LINE_COLOR * (outline + glow * 2.0);
}

// Orbiting shapes - unrolled
vec3 orbitingShapes(vec2 uv, vec2 center, float atime) {
    vec3 color = vec3(0.0);
    float orbitRadius = 0.15;
    
    float angle0 = atime * 0.5 + 0.0 * PI * 2.0 / 3.0;
    vec2 orbitPos0 = center + vec2(cos(angle0), sin(angle0)) * orbitRadius;
    vec2 p0 = uv - orbitPos0;
    float d0 = length(p0) - 0.015;
    float outline0 = smoothstep(0.002, 0.0, abs(d0));
    float glow0 = smoothstep(0.02, 0.0, abs(d0)) * 0.5;
    color += LINE_COLOR * (outline0 + glow0);
    
    float angle1 = atime * 0.5 + 1.0 * PI * 2.0 / 3.0;
    vec2 orbitPos1 = center + vec2(cos(angle1), sin(angle1)) * orbitRadius;
    vec2 p1 = uv - orbitPos1;
    float d1 = length(p1) - 0.015;
    float outline1 = smoothstep(0.002, 0.0, abs(d1));
    float glow1 = smoothstep(0.02, 0.0, abs(d1)) * 0.5;
    color += LINE_COLOR * (outline1 + glow1);
    
    float angle2 = atime * 0.5 + 2.0 * PI * 2.0 / 3.0;
    vec2 orbitPos2 = center + vec2(cos(angle2), sin(angle2)) * orbitRadius;
    vec2 p2 = uv - orbitPos2;
    float d2 = length(p2) - 0.015;
    float outline2 = smoothstep(0.002, 0.0, abs(d2));
    float glow2 = smoothstep(0.02, 0.0, abs(d2)) * 0.5;
    color += LINE_COLOR * (outline2 + glow2);
    
    return color;
}

// Scanlines
float scanlines(vec2 uv, float atime) {
    float line = sin((uv.y + atime * SCANLINE_SPEED) * 200.0);
    return smoothstep(0.9, 1.0, line);
}

// Helper function: signed distance to line segment
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}


// W symbol with glowing blue effect
vec3 letter_W(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the W shape using line segments
    // W has 4 diagonal strokes forming the letter
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Left outer stroke (top-left to bottom-middle-left)
    vec2 p1_start = vec2(-0.5, 0.5);
    vec2 p1_end = vec2(-0.25, -0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Left inner stroke (bottom-middle-left to top-middle)
    vec2 p2_start = vec2(-0.25, -0.5);
    vec2 p2_end = vec2(0.0, 0.2);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Right inner stroke (top-middle to bottom-middle-right)
    vec2 p3_start = vec2(0.0, 0.2);
    vec2 p3_end = vec2(0.25, -0.5);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Right outer stroke (bottom-middle-right to top-right)
    vec2 p4_start = vec2(0.25, -0.5);
    vec2 p4_end = vec2(0.5, 0.5);
    float d4 = sdSegment(p, p4_start, p4_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), min(d3, d4));
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// A symbol with glowing blue effect
vec3 letter_A(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the A shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Left stroke (bottom-left to top-center)
    vec2 p1_start = vec2(-0.4, -0.5);
    vec2 p1_end = vec2(0.0, 0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Right stroke (top-center to bottom-right)
    vec2 p2_start = vec2(0.0, 0.5);
    vec2 p2_end = vec2(0.4, -0.5);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Horizontal crossbar (middle of both strokes)
    vec2 p3_start = vec2(-0.2, 0.0);
    vec2 p3_end = vec2(0.2, 0.0);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), d3);
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// Y symbol with glowing blue effect
vec3 letter_Y(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the Y shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Left stroke (top-left to center)
    vec2 p1_start = vec2(-0.4, 0.5);
    vec2 p1_end = vec2(0.0, 0.0);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Right stroke (top-right to center)
    vec2 p2_start = vec2(0.4, 0.5);
    vec2 p2_end = vec2(0.0, 0.0);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Vertical stem (center to bottom)
    vec2 p3_start = vec2(0.0, 0.0);
    vec2 p3_end = vec2(0.0, -0.5);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), d3);
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// F symbol with glowing blue effect
// F symbol with glowing blue effect (flipped)
vec3 letter_F(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the F shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Vertical stroke (bottom to top)
    vec2 p1_start = vec2(0.3, -0.5);
    vec2 p1_end = vec2(0.3, 0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Top horizontal stroke (right to left)
    vec2 p2_start = vec2(0.3, 0.5);
    vec2 p2_end = vec2(-0.3, 0.5);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Middle horizontal stroke (right to left, shorter)
    vec2 p3_start = vec2(0.3, 0.05);
    vec2 p3_end = vec2(-0.2, 0.05);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), d3);
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// I symbol with glowing blue effect
vec3 letter_I(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the I shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Vertical stroke (bottom to top)
    vec2 p1_start = vec2(0.0, -0.5);
    vec2 p1_end = vec2(0.0, 0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Top horizontal stroke (serif)
    vec2 p2_start = vec2(-0.25, 0.5);
    vec2 p2_end = vec2(0.25, 0.5);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Bottom horizontal stroke (serif)
    vec2 p3_start = vec2(-0.25, -0.5);
    vec2 p3_end = vec2(0.25, -0.5);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), d3);
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// R symbol with glowing blue effect (flipped)
vec3 letter_R(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the R shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Vertical stroke (bottom to top)
    vec2 p1_start = vec2(0.3, -0.5);
    vec2 p1_end = vec2(0.3, 0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Top horizontal stroke
    vec2 p2_start = vec2(0.3, 0.5);
    vec2 p2_end = vec2(-0.25, 0.5);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Left vertical stroke (top bowl)
    vec2 p3_start = vec2(-0.25, 0.5);
    vec2 p3_end = vec2(-0.25, 0.05);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Middle horizontal stroke (bottom of bowl)
    vec2 p4_start = vec2(-0.25, 0.05);
    vec2 p4_end = vec2(0.3, 0.05);
    float d4 = sdSegment(p, p4_start, p4_end);
    
    // Diagonal leg (middle to bottom-left)
    vec2 p5_start = vec2(0.05, 0.05);
    vec2 p5_end = vec2(-0.3, -0.5);
    float d5 = sdSegment(p, p5_start, p5_end);
    
    // Combine all strokes
    float d = min(min(min(d1, d2), min(d3, d4)), d5);
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

// E symbol with glowing blue effect
vec3 letter_E(vec2 uv, vec2 center, float size) {
    vec2 p = (uv - center) / size;
    
    vec3 color = vec3(0.0);
    
    // Define the E shape using line segments
    float line_width = 0.08;
    float glow_width = 0.15;
    
    // Vertical stroke (bottom to top)
    vec2 p1_start = vec2(0.3, -0.5);
    vec2 p1_end = vec2(0.3, 0.5);
    float d1 = sdSegment(p, p1_start, p1_end);
    
    // Top horizontal stroke (right to left)
    vec2 p2_start = vec2(0.3, 0.5);
    vec2 p2_end = vec2(-0.3, 0.5);
    float d2 = sdSegment(p, p2_start, p2_end);
    
    // Middle horizontal stroke (right to left)
    vec2 p3_start = vec2(0.3, 0.0);
    vec2 p3_end = vec2(-0.2, 0.0);
    float d3 = sdSegment(p, p3_start, p3_end);
    
    // Bottom horizontal stroke (right to left)
    vec2 p4_start = vec2(0.3, -0.5);
    vec2 p4_end = vec2(-0.3, -0.5);
    float d4 = sdSegment(p, p4_start, p4_end);
    
    // Combine all strokes
    float d = min(min(d1, d2), min(d3, d4));
    
    // Create the main line
    float line = smoothstep(line_width, line_width * 0.5, d);
    
    // Create the glow
    float glow = smoothstep(glow_width, 0.0, d) * 0.5;
    
    // Apply blue color
    color = LINE_COLOR * (line * 2.0 + glow);
    
    return color;
}

void main() {
    // STAR NEST BACKGROUND
    vec2 uv = v_uv - 0.5;
    uv.y *= u_resolution.y / u_resolution.x;
    vec3 dir = vec3(uv * zoom, 1.0);
    
    float time = u_time * speed + 0.25;
    
    float a1 = 0.5 + sin(u_time * 0.001) * 0.5;
    float a2 = 0.8 + cos(u_time * 0.0015) * 0.5;
    mat2 rot1 = mat2(cos(a1), sin(a1), -sin(a1), cos(a1));
    mat2 rot2 = mat2(cos(a2), sin(a2), -sin(a2), cos(a2));
    dir.xz *= rot1;
    dir.xy *= rot2;
    
    vec3 from = vec3(1.0, 0.5, 0.5);
    from += vec3(time * 2.0, time, -2.0);
    from.xz *= rot1;
    from.xy *= rot2;
    
    float s = 0.1;
    float fade = 1.0;
    vec3 v = vec3(0.0);
    
    for (int r = 0; r < volsteps; r++) {
        vec3 p = from + s * dir * 10.5;
        p = abs(vec3(tile) - mod(p, vec3(tile * 2.0)));
        
        float pa = 0.0;
        float a = 0.0;
        for (int i = 0; i < iterations; i++) { 
            p = abs(p) / dot(p, p) - formuparam;
            a += abs(length(p) - pa);
            pa = length(p);
        }
        
        float dm = max(0.0, darkmatter - a * a * 0.1);
        a *= a * a;
        
        fade *= 1.0 - dm;
        v += fade;
        v += vec3(s, s * s, s * s * s) * a * brightness * fade;
        fade *= distfading;
        s += stepsize;
    }
    
    v = mix(vec3(length(v)), v, saturation);
    vec3 finalColor = v * 0.01;
    
    // NEW TRON ELEMENTS
    vec2 centered_uv = v_uv - 0.5;
    centered_uv.y *= u_resolution.y / u_resolution.x;
    
    // Add subtle grid
    float gridPattern = grid(centered_uv * 5.0);
    finalColor += LINE_COLOR * gridPattern * GRID_GLOW * 0.3;
    
    // Add horizontal scanlines
    float scan = scanlines(v_uv, u_time);
    finalColor += vec3(0.0, 0.3, 0.6) * scan * 0.15;
    
    // Add vertical accent lines (sparse)
    float verticalLines = smoothstep(0.002, 0.0, abs(fract(v_uv.x * 8.0) - 0.5));
    finalColor += LINE_COLOR * verticalLines * 0.2 * sin(u_time * 0.5 + v_uv.x * 10.0) * 0.5;
    
    // ANIMATED SHAPES - Multiple spinning hexagons
   /* finalColor += spinningHexagon(centered_uv, vec2(-0.3, 0.2), 0.08, u_time, 1.0);
    finalColor += spinningHexagon(centered_uv, vec2(0.35, -0.15), 0.06, u_time, -1.5);
    finalColor += spinningHexagon(centered_uv, vec2(-0.25, -0.25), 0.05, u_time, 0.8);
    finalColor += spinningHexagon(centered_uv, vec2(0.2, 0.25), 0.07, u_time, -1.2);
    */
    // CENTER PINWHEEL (replaces rotating ring)
 /*   finalColor += pinwheel(centered_uv, vec2(0.0, 0.0), 0.04, u_time, 0.6);
 
vec3 w_symbol = letter_W(uv, vec2(-0.2, 0.0), -0.05); // position at (0.5, 0.3), size 0.15
finalColor += w_symbol;

vec3 a_symbol = letter_A(uv, vec2(-0.15, 0.0), -0.05);
finalColor += a_symbol;
 
// Add this where you want to render the Y symbol
vec3 y_symbol = letter_Y(uv, vec2(-0.1,  0.0), -0.05);// position at (0.9, 0.3), size 0.15
finalColor += y_symbol;

vec3 f_symbol = letter_F(uv,  vec2(0.1,  0.0), -0.05);
finalColor += f_symbol;

vec3 i_symbol = letter_I(uv,  vec2(0.15,  0.0), -0.05);
finalColor += i_symbol;

vec3 r_symbol = letter_R(uv, vec2(0.2,  0.0), -0.05);
finalColor += r_symbol;

vec3 e_symbol = letter_E(uv, vec2(0.25,  0.0), -0.05);
finalColor += e_symbol;
    */

    // Rotating triangles
//    finalColor += rotatingTriangle(centered_uv, vec2(-0.35, -0.05), 0.06, u_time, -1.0);
//    finalColor += rotatingTriangle(centered_uv, vec2(0.15, -0.3), 0.05, u_time, 1.3);
    
    // Pulsing hexagons
//    finalColor += pulsingHexagon(centered_uv, vec2(-0.15, 0.15), 0.055, u_time, 2.0);
  //  finalColor += pulsingHexagon(centered_uv, vec2(0.3, 0.0), 0.045, u_time, 2.5);
    
    // Orbiting small shapes around center
   // finalColor += orbitingShapes(centered_uv, vec2(0.0, 0.0), u_time);
    
    // TRON TRAVELING LINES OVERLAY (Original)
    float edge_dist = distanceToEdge(v_uv, u_resolution);
    
    if (edge_dist < 0.05 && edge_dist > 0.0) {
        float perimeter_pos = uvToPerimeter(v_uv, u_resolution);
        float perimeter_length = 2.0 * (1.0 + (u_resolution.y - 2.0 * EDGE_MARGIN) / (u_resolution.x - 2.0 * EDGE_MARGIN));
        float spacing = perimeter_length / float(NUM_LINES);
        
        float edge_falloff = smoothstep(0.05, 0.0, edge_dist);
        float sharp_line = smoothstep(LINE_WIDTH * 2.0, 0.0, edge_dist);
        float soft_glow = smoothstep(GLOW_WIDTH, 0.0, edge_dist);
        
        for (int i = 0; i < NUM_LINES; i++) {
            float line_pos = mod(u_time * ANIMATION_SPEED + float(i) * spacing, perimeter_length);
            float dist_to_line = abs(perimeter_pos * perimeter_length - line_pos);
            dist_to_line = min(dist_to_line, abs(dist_to_line - perimeter_length));
            
            float line_intensity = smoothstep(LINE_LENGTH, 0.0, dist_to_line);
            float trail_intensity = smoothstep(LINE_LENGTH * 2.0, LINE_LENGTH, dist_to_line) * 0.3;
            line_intensity = max(line_intensity, trail_intensity);
            
            vec3 glow_color = LINE_COLOR * 0.6;
            
            finalColor += LINE_COLOR * sharp_line * line_intensity * edge_falloff * 2.0;
            finalColor += glow_color * soft_glow * line_intensity * edge_falloff;
        }
        
        finalColor += vec3(0.3, 0.4, 0.5) * edge_falloff * BASE_EDGE_GLOW * 0.3;
    }
    
    gl_FragColor = vec4(finalColor, 1.0);
}
)";


static const char *cursor_vertex_shader = R"(
#version 100
attribute vec2 position;
uniform mat4 mvp;
varying vec2 v_pos;

void main() {
    v_pos = position;
    gl_Position = mvp * vec4(position.x, position.y, 0.001, 1.0);
}
)";

static const char *cursor_fragment_shader = R"(
#version 100
precision mediump float;
varying vec2 v_pos;
uniform float u_time;
uniform vec3 u_color;

void main() {
    float dist = length(v_pos);
    
    // Create a ring with animated pulse
    float ring_outer = 0.08;
    float ring_inner = 0.06;
    float pulse = sin(u_time * 4.0) * 0.01 + 0.07;
    
    // Ring alpha
    float ring_alpha = smoothstep(ring_outer, ring_inner, dist) * 
                      smoothstep(pulse - 0.02, pulse, dist);
    
    // Center dot
    float dot_alpha = smoothstep(0.02, 0.01, dist);
    
    float alpha = max(ring_alpha, dot_alpha);
    
    // Bright color with glow
    vec3 color = u_color * (1.0 + sin(u_time * 3.0) * 0.3);
    
    gl_FragColor = vec4(color, alpha * 0.8);
}
)";

   // NEW: Beam shaders
    const char *beam_vertex_shader = R"(
#version 100
attribute vec3 position;
attribute float distance;  // Distance from beam center (0 to 1)
uniform mat4 mvp;
varying float v_distance;

void main() {
    v_distance = distance;
    gl_Position = mvp * vec4(position, 1.0);
}
)";
    const char *beam_fragment_shader = R"(
#version 100
#ifdef GL_ES
precision mediump float;
#endif

uniform vec3 color;
varying float v_distance;  // Distance from beam center

void main() {
    // Clamp distance to prevent artifacts
    float dist = clamp(v_distance, 0.0, 1.0);
    
    // Create radial falloff from center of beam
    float glow = 1.0 - dist;
    
    // Apply smooth falloff curve for softer glow
    glow = pow(glow, 2.5);
    
    // Add bright core in the center
    float core = 1.0 - smoothstep(0.0, 0.15, dist);
    core = pow(core, 3.0);
    
    // Combine glow with brighter core
    float intensity = glow + core * 2.0;
    
    // Make the color brighter in the center
    vec3 finalColor = color * (1.0 + core * 0.5);
    
    // Clamp final intensity to prevent overflow
    intensity = clamp(intensity, 0.0, 2.0);
    
    gl_FragColor = vec4(finalColor, intensity * 0.8);
}
)";