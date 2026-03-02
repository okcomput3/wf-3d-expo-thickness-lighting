/* ═══════════════════════════════════════════════════════════════════════════
 *  FLAT GRID EXPO — 3D BOX MESH VERSION v6
 *
 *  Based on v5 performance-patched version.
 *  NEW: Each workspace is rendered as a 3D cuboid mesh:
 *    - Front face: desktop FBO texture
 *    - Back + 4 side faces: multi-hue colored with lighting
 *    - Depth testing for proper 3D occlusion
 *    - Per-workspace unique color via golden-ratio HSV distribution
 *    - Gradient blending between two hues per face
 * ═══════════════════════════════════════════════════════════════════════════*/

#include <wayfire/per-output-plugin.hpp>
#include <memory>
#include <wayfire/plugin.hpp>
#include <wayfire/opengl.hpp>
#include <wayfire/output.hpp>
#include <wayfire/core.hpp>
#include <wayfire/workspace-stream.hpp>
#include <wayfire/render-manager.hpp>
#include <wayfire/workspace-set.hpp>
#include <wayfire/scene-operations.hpp>
#include <wayfire/plugins/common/input-grab.hpp>
#include "wayfire/plugins/ipc/ipc-activator.hpp"
#include <linux/input-event-codes.h>
#include "wayfire/plugins/wobbly/wobbly-signal.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <glm/gtc/matrix_inverse.hpp>
#include <wayfire/img.hpp>

#include "cube.hpp"
#include "simple-background.hpp"
#include "skydome.hpp"
#include "cubemap.hpp"
#include "cube-control-signal.hpp"
#include "wayfire/region.hpp"
#include "wayfire/scene-render.hpp"
#include "wayfire/scene.hpp"
#include "wayfire/signal-definitions.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

#ifdef USE_GLES32
    #include <GLES3/gl32.h>
#endif

#include "shaders.tpp"
#include "shaders-3-2.tpp"

/* ─── constants ─────────────────────────────────────────────────────────── */
#define Z_OFFSET_NEAR  0.89567f
#define Z_OFFSET_FAR   2.00000f
#define ZOOM_MAX       10.0f
#define ZOOM_MIN       0.1f
#define GRID_GAP       0.15f
#define PANEL_SCALE    1.00f
#define HIGHLIGHT_PAD  0.015f
#define CULL_PX        8.0f

/* 3D BOX DEPTH — thickness of each workspace cuboid */
#define BOX_DEPTH      0.32f

/* OPT B: max FBO renders per frame (staggered) */
#define MAX_FBO_RENDERS_PER_FRAME 999

/* OPT A: LOD thresholds */
#define LOD_MIN_SCALE   0.50f
#define LOD_MAX_SCALE   1.00f
#define LOD_HYSTERESIS  0.08f

/* OPT M: pointer dedup threshold (squared pixels) */
#define POINTER_DEDUP_DIST_SQ  0.5f

#define CORNER_RADIUS   0.06f
#define CORNER_SEGMENTS 8

/* ─── HSV to RGB helper ─────────────────────────────────────────────────── */
static glm::vec3 hsv2rgb(float h, float s, float v)
{
    h = std::fmod(h, 1.0f);
    if (h < 0.0f) h += 1.0f;
    float c  = v * s;
    float hp = h * 6.0f;
    float x  = c * (1.0f - std::abs(std::fmod(hp, 2.0f) - 1.0f));
    float m  = v - c;
    glm::vec3 rgb;
    if      (hp < 1.0f) rgb = {c, x, 0};
    else if (hp < 2.0f) rgb = {x, c, 0};
    else if (hp < 3.0f) rgb = {0, c, x};
    else if (hp < 4.0f) rgb = {0, x, c};
    else if (hp < 5.0f) rgb = {x, 0, c};
    else                rgb = {c, 0, x};
    return rgb + glm::vec3(m);
}

/* Generate two complementary hues for a workspace */
static void workspace_colors(int wx, int wy, int gw,
                             glm::vec3 &color1, glm::vec3 &color2)
{
    int idx = wy * gw + wx;
    float base_hue = std::fmod(idx * 0.618033988f, 1.0f);
    color1 = hsv2rgb(base_hue, 0.6f, 0.7f);
    color2 = hsv2rgb(base_hue + 0.15f, 0.5f, 0.55f);
}

/* ─── Shader sources ────────────────────────────────────────────────────── */
static const char *bg_vs = R"(
#version 100
attribute vec2 position;
varying vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.999, 1.0);
})";
static const char *bg_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform float u_time;
void main() {
    vec3 top    = vec3(0.08, 0.08, 0.14);
    vec3 bottom = vec3(0.03, 0.03, 0.06);
    vec3 c = mix(bottom, top, v_uv.y);
    float vig = 1.0 - length(v_uv - 0.5) * 0.5;
    c *= vig;
    gl_FragColor = vec4(c, 1.0);
})";

/* Front face: textured with FBO */
static const char *panel_vs = R"(
#version 100
attribute vec2 position;
attribute vec2 uvPosition;
uniform mat4 MVP;
varying vec2 v_uv;
void main() {
    v_uv = uvPosition;
    gl_Position = MVP * vec4(position, 0.005, 1.0);
})";
static const char *panel_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D tex;
uniform float u_brightness;
void main() {
    vec4 t = texture2D(tex, v_uv);
    gl_FragColor = vec4(t.rgb * u_brightness, t.a);
})";
/* 3D BOX FACES: colored sides and back with lighting + gradient */
static const char *box_color_vs = R"(
#version 100
attribute vec3 a_position;
attribute vec3 a_normal;
uniform mat4 MVP;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;
varying vec3 v_normal_ws;
varying vec3 v_pos_ws;
varying vec3 v_pos_local;
void main() {
    v_pos_local = a_position;
    v_pos_ws = (u_model * vec4(a_position, 1.0)).xyz;
    v_normal_ws = normalize(u_normal_matrix * a_normal);
    gl_Position = MVP * vec4(a_position, 1.0);
})";
static const char *box_color_fs = R"(
#version 100
precision highp float;
varying vec3 v_normal_ws;
varying vec3 v_pos_ws;
varying vec3 v_pos_local;

uniform vec3  u_color1;
uniform vec3  u_color2;
uniform float u_brightness;

/* Dynamic lights */
uniform vec3  u_cam_pos;        /* camera world position         */
uniform vec3  u_light_pos;      /* primary light (follows cam)   */
uniform vec3  u_light_color;    /* primary light color            */
uniform vec3  u_accent_pos;     /* accent light (selected ws)    */
uniform vec3  u_accent_color;   /* accent light color             */
uniform float u_time;

void main() {
    /* Base color: gradient blend on local Y */
    float blend = v_pos_local.y + 0.5;
    vec3 base_color = mix(u_color1, u_color2, blend);

    vec3 N = normalize(v_normal_ws);
    vec3 V = normalize(u_cam_pos - v_pos_ws);

    /* ── Ambient ─────────────────────────────────────── */
    float ambient = 0.15;

    /* ── Primary light (camera-tracking) ─────────────── */
    vec3  L1     = u_light_pos - v_pos_ws;
    float dist1  = length(L1);
    L1 = normalize(L1);
    float atten1 = 1.0 / (1.0 + 0.05 * dist1 + 0.01 * dist1 * dist1);
    float diff1  = max(dot(N, L1), 0.0);
    vec3  H1     = normalize(L1 + V);
    float spec1  = pow(max(dot(N, H1), 0.0), 32.0);

    /* ── Accent light (selected workspace) ───────────── */
    vec3  L2     = u_accent_pos - v_pos_ws;
    float dist2  = length(L2);
    L2 = normalize(L2);
    float atten2 = 1.0 / (1.0 + 0.08 * dist2 + 0.02 * dist2 * dist2);
    float diff2  = max(dot(N, L2), 0.0);
    vec3  H2     = normalize(L2 + V);
    float spec2  = pow(max(dot(N, H2), 0.0), 24.0);

    /* ── Subtle pulse on accent light ────────────────── */
    float pulse = 0.85 + 0.15 * sin(u_time * 2.0);

    /* ── Rim light (edge glow facing camera) ─────────── */
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.2;

    /* ── Combine ─────────────────────────────────────── */
    vec3 lighting = vec3(ambient);
    lighting += (diff1 * 0.6 + spec1 * 0.3) * atten1 * u_light_color;
    lighting += (diff2 * 0.4 + spec2 * 0.2) * atten2 * u_accent_color * pulse;
    lighting += rim * vec3(1.0);

    vec3 col = base_color * lighting * u_brightness;

    /* Subtle tone mapping to avoid blowout */
    col = col / (col + vec3(1.0));

    gl_FragColor = vec4(col, 1.0);
})";
/* OPT C: simple fullscreen blit shader */
static const char *blit_vs = R"(
#version 100
attribute vec2 position;
varying vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.999, 1.0);
})";
static const char *blit_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D tex;
void main() {
    gl_FragColor = texture2D(tex, v_uv);
})";

static const char *ring_vs = R"(
#version 100
attribute vec2 position;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(position, 0.0, 1.0); }
)";
static const char *ring_fs = R"(
#version 100
precision highp float;
uniform float u_time;
uniform vec3  u_color;
void main() {
    float p = sin(u_time * 4.0) * 0.2 + 0.8;
    gl_FragColor = vec4(u_color * p, 0.55 * p);
})";


/* ─── Glowing light orb billboard ───────────────────────────────────── */
static const char *orb_vs = R"(
#version 100
attribute vec2 position;
uniform mat4 MVP;
uniform float u_size;
varying vec2 v_uv;
void main() {
    v_uv = position;
    gl_Position = MVP * vec4(position * u_size, 0.0, 1.0);
})";
static const char *orb_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform vec3  u_color;
uniform float u_time;
uniform float u_intensity;
void main() {
    float d = length(v_uv);

    /* Hot white core */
    float core = exp(-d * 8.0) * 1.5;

    /* Soft inner glow */
    float glow = exp(-d * 3.0) * 0.8;

    /* Outer halo with pulse */
    float pulse = 0.9 + 0.1 * sin(u_time * 3.0);
    float halo = exp(-d * 1.2) * 0.4 * pulse;

    /* Rays/spikes (4-pointed star) */
    float angle = atan(v_uv.y, v_uv.x);
    float spike = pow(abs(cos(angle * 2.0)), 32.0) * exp(-d * 2.5) * 0.6;
    spike += pow(abs(cos(angle * 2.0 + 0.785)), 32.0) * exp(-d * 3.0) * 0.3;

    float total = (core + glow + halo + spike) * u_intensity;

    vec3 col = mix(u_color, vec3(1.0), core * 0.7);
    col = col * total;

    gl_FragColor = vec4(col, total * 0.9);
})";

/* ─── God rays (screen-space radial blur) ───────────────────────────── */
static const char *godrays_vs = R"(
#version 100
attribute vec2 position;
varying vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
})";
static const char *godrays_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_scene;
uniform vec2  u_light_screen;   /* light pos in UV [0,1] space */
uniform float u_density;
uniform float u_weight;
uniform float u_decay;
uniform float u_exposure;
uniform float u_intensity;

const int NUM_SAMPLES = 48;

void main() {
    vec2 dir = v_uv - u_light_screen;
    dir *= 1.0 / float(NUM_SAMPLES) * u_density;

    vec2 tc = v_uv;
    vec3 color = vec3(0.0);
    float illumination_decay = 1.0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        tc -= dir;
        vec3 s = texture2D(u_scene, clamp(tc, 0.0, 1.0)).rgb;

        /* Only gather bright pixels */
        float lum = dot(s, vec3(0.299, 0.587, 0.114));
        s *= smoothstep(0.3, 0.8, lum);

        s *= illumination_decay * u_weight;
        color += s;
        illumination_decay *= u_decay;
    }

    color *= u_exposure * u_intensity;
    gl_FragColor = vec4(color, 1.0);
})";

/* ─── Bloom extract + blur ──────────────────────────────────────────── */
static const char *bloom_extract_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_scene;
uniform float u_threshold;
void main() {
    vec3 c = texture2D(u_scene, v_uv).rgb;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float contrib = max(0.0, lum - u_threshold);
    c *= contrib / (lum + 0.001);
    gl_FragColor = vec4(c, 1.0);
})";

static const char *blur_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_direction;  /* (1/w, 0) or (0, 1/h) */

void main() {
    vec3 color = vec3(0.0);
    /* 9-tap Gaussian */
    color += texture2D(u_tex, v_uv + u_direction * -4.0).rgb * 0.0162;
    color += texture2D(u_tex, v_uv + u_direction * -3.0).rgb * 0.0540;
    color += texture2D(u_tex, v_uv + u_direction * -2.0).rgb * 0.1216;
    color += texture2D(u_tex, v_uv + u_direction * -1.0).rgb * 0.1945;
    color += texture2D(u_tex, v_uv).rgb                      * 0.2270;
    color += texture2D(u_tex, v_uv + u_direction *  1.0).rgb * 0.1945;
    color += texture2D(u_tex, v_uv + u_direction *  2.0).rgb * 0.1216;
    color += texture2D(u_tex, v_uv + u_direction *  3.0).rgb * 0.0540;
    color += texture2D(u_tex, v_uv + u_direction *  4.0).rgb * 0.0162;
    gl_FragColor = vec4(color, 1.0);
})";

/* ─── Final compositing (scene + godrays + bloom) ───────────────────── */
static const char *composite_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_godrays;
uniform sampler2D u_bloom;
uniform sampler2D u_occlude;
uniform float u_godray_strength;
uniform float u_bloom_strength;
uniform float u_vignette;
void main() {
    vec3 scene = texture2D(u_scene, v_uv).rgb;
    vec3 rays  = texture2D(u_godrays, v_uv).rgb;
    vec3 bloom = texture2D(u_bloom, v_uv).rgb;

    /* Use occlusion map as mask: black = cube area, bright = gaps/bg */
    /* Where cubes are (black in occlude), effects are suppressed */
    float occlude = texture2D(u_occlude, v_uv).r;
    float mask = smoothstep(0.0, 0.15, occlude);

    /* Only apply effects where there are no cubes */
    vec3 col = scene;
    col += rays * u_godray_strength * mask;
    col += bloom * u_bloom_strength * mask;

    /* Only tone map and vignette the non-desktop areas */
    vec3 tonemapped = col / (col + vec3(0.8));
    tonemapped = pow(tonemapped, vec3(1.0 / 2.2));

    /* Blend: desktop pixels stay raw, background gets effects */
    col = mix(scene, tonemapped, mask);

    /* Vignette — subtle, applied everywhere is fine */
    float vig = 1.0 - length(v_uv - 0.5) * u_vignette;
    vig = clamp(vig * vig, 0.0, 1.0);
    col *= mix(1.0, vig, mask * 0.5 + 0.5);

    gl_FragColor = vec4(col, 1.0);
})";
/* ─── Neon edge glow on selected box ────────────────────────────────── */
static const char *neon_edge_vs = R"(
#version 100
attribute vec3 a_position;
attribute vec3 a_normal;
uniform mat4 MVP;
uniform float u_expand;
void main() {
    /* Push vertices outward along their normal to create outline */
    vec3 expanded = a_position + a_normal * u_expand;
    gl_Position = MVP * vec4(expanded, 1.0);
})";
static const char *neon_edge_fs = R"(
#version 100
precision highp float;
uniform vec3  u_color;
uniform float u_time;
uniform float u_intensity;
void main() {
    float pulse = 0.7 + 0.3 * sin(u_time * 4.0);
    vec3 col = u_color * u_intensity * pulse;
    /* Bright edge with soft alpha */
    gl_FragColor = vec4(col, 0.8 * pulse);
})";

/* Render geometry as solid black (for occlusion) */
static const char *silhouette_vs = R"(
#version 100
attribute vec3 a_position;
uniform mat4 MVP;
void main() {
    vec3 pos = a_position;
    if (pos.z > -0.001) pos.z = 0.005;
    gl_Position = MVP * vec4(pos, 1.0);
})";
static const char *silhouette_fs = R"(
#version 100
precision highp float;
void main() {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
})";

/* Big soft light disc for the occlusion pass */
static const char *light_disc_vs = R"(
#version 100
attribute vec2 position;
uniform mat4 MVP;
uniform float u_size;
varying vec2 v_uv;
void main() {
    v_uv = position;
    gl_Position = MVP * vec4(position * u_size, 0.0, 1.0);
})";
static const char *light_disc_fs = R"(
#version 100
precision highp float;
varying vec2 v_uv;
uniform vec3  u_color;
uniform float u_time;
void main() {
    float d = length(v_uv);

    /* Discard pixels outside the circle */
    if (d > 0.1) discard;

    /* Large soft bright disc */
    float core = exp(-d * 2.0) * 2.0;
    float glow = exp(-d * 0.8) * 1.0;
    float outer = exp(-d * 0.3) * 0.4;

    /* Smooth fade to zero at edge */
    float edge = 1.0 - smoothstep(0.7, 1.0, d);

    /* Subtle pulse */
    float pulse = 0.9 + 0.1 * sin(u_time * 2.5);

    float total = (core + glow + outer) * pulse * edge;
    vec3 col = u_color * total;

    gl_FragColor = vec4(col, total);
})";
/* ═══════════════════════════════════════════════════════════════════════════
 *  SCREEN-SIZE HELPER  (draw-call culling + LOD)
 * ═══════════════════════════════════════════════════════════════════════════*/
static float projected_quad_size_px(const glm::mat4& mvp,
                                    float out_w, float out_h)
{
    auto proj = [&](float x, float y) -> glm::vec2 {
        glm::vec4 c = mvp * glm::vec4(x, y, 0.0f, 1.0f);
        if (std::abs(c.w) < 1e-6f) return {0,0};
        return { (c.x/c.w*0.5f+0.5f)*out_w,
                 (c.y/c.w*0.5f+0.5f)*out_h };
    };
    glm::vec2 tl = proj(-0.5f, 0.5f);
    glm::vec2 tr = proj( 0.5f, 0.5f);
    glm::vec2 bl = proj(-0.5f,-0.5f);
    return std::max(glm::length(tr-tl), glm::length(bl-tl));
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN PLUGIN CLASS
 * ═══════════════════════════════════════════════════════════════════════════*/

class wayfire_cube : public wf::per_output_plugin_instance_t,
                     public wf::pointer_interaction_t
{
    std::chrono::steady_clock::time_point last_frame_time =
        std::chrono::steady_clock::now();
    float elapsed = 0.0f;

    wf::point_t selected_ws = {0, 0};
    wf::point_t origin_ws   = {0, 0};

    /* FIX 1: track previous selection */
    wf::point_t prev_selected_ws = {-1, -1};

    bool is_dragging_window = false;
    wayfire_toplevel_view dragged_view = nullptr;
    wf::pointf_t drag_start_cursor;
    wf::point_t  drag_start_workspace;
    wf::pointf_t last_cursor_pos;
    wf::pointf_t drag_offset;
    bool workspace_already_set = false;
    int startup_frames = 0;
    wf::point_t right_click_target_ws = {-1, -1};
    int dragged_window_face_index = -1;
     int skip_first_render = 0;

    struct CursorIndicator {
        bool active = false;
        glm::vec3 world_position;
        glm::vec2 uv_position;
        int workspace_x = -1, workspace_y = -1;
        float timestamp = 0.0f;
    } cursor_indicator;

    bool sync_cam_y_to_z = false;
    float plane_z = 0.0f;
    bool has_virtual_hit = false;
    glm::vec3 virtual_ray_hit_pos{0};
    int hit_workspace_x = -1, hit_workspace_y = -1;

       int front_index_count = 0;
    int sides_index_count = 0;

    wf::animation::simple_animation_t popout_scale_animation{wf::create_option<int>(300)};
    wf::animation::simple_animation_t camera_y_offset{wf::create_option<int>(600)};

    enum class ZoomState { OVERVIEW, ZOOMING_IN, ZOOMED_IN, ZOOMING_OUT };
    ZoomState zoom_state = ZoomState::OVERVIEW;

    struct ZoomAnim {
        bool running = false;
        float progress = 0.0f, duration = 0.5f;
        glm::vec3 start_pos{0}, end_pos{0}, current_pos{0};
        static float ease(float t) {
            t = std::clamp(t,0.0f,1.0f);
            return t<0.5f ? 4*t*t*t : 1.0f-std::pow(-2*t+2,3)/2.0f;
        }
        void start(glm::vec3 from, glm::vec3 to, float dur=0.5f) {
            start_pos=from; end_pos=to; current_pos=from;
            progress=0.0f; duration=dur; running=true;
        }
        bool update(float dt) {
            if (!running) return false;
            progress += dt/duration;
            if (progress>=1.0f) {
                progress=1.0f; running=false; current_pos=end_pos; return false;
            }
            current_pos = glm::mix(start_pos, end_pos, ease(progress));
            return true;
        }
        void snap(glm::vec3 pos) {
            running=false; progress=1.0f;
            start_pos=end_pos=current_pos=pos;
        }
    } zoom_anim;
GLuint pp_occlude_tex = 0, pp_occlude_fbo = 0;
    wf::point_t zoom_target_ws = {0, 0};
    /* ═══ POST-PROCESSING FBOs ═══ */
    GLuint pp_scene_depth_rbo = 0;
    GLuint pp_scene_tex = 0, pp_scene_fbo = 0;
    GLuint pp_godrays_tex = 0, pp_godrays_fbo = 0;
    GLuint pp_bloom_tex[2] = {0, 0};
    GLuint pp_bloom_fbo[2] = {0, 0};
    int pp_width = 0, pp_height = 0;

    OpenGL::program_t godrays_program;
    OpenGL::program_t bloom_extract_program;
    OpenGL::program_t blur_program;
    OpenGL::program_t composite_program;
    OpenGL::program_t orb_program;
    OpenGL::program_t neon_edge_program;
    OpenGL::program_t silhouette_program;
    OpenGL::program_t light_disc_program;

    GLuint cursor_vbo = 0, background_vbo = 0;
    GLuint panel_vbo = 0;   /* OPT D: front face quad */
    GLuint panel_ibo = 0;   /* OPT E: front face indices */

    /* ═══ 3D BOX GEOMETRY ═══ */
    GLuint box_sides_vbo = 0;   /* back + 4 side faces (pos3 + normal3) */
    GLuint box_sides_ibo = 0;   /* indices for 5 faces × 2 triangles */

    OpenGL::program_t program, cap_program, background_program,
                      beam_program, cursor_program;
    OpenGL::program_t blit_program;  /* OPT C */
    OpenGL::program_t box_color_program;  /* 3D box coloured faces */

    static constexpr GLfloat panel_verts[]   = {-0.5f,0.5f,0.5f,0.5f,0.5f,-0.5f,-0.5f,-0.5f};
    static constexpr GLfloat panel_uvs[]     = {0,1,1,1,1,0,0,0};

    /* OPT C: cached background texture */
    GLuint bg_cache_tex = 0;
    GLuint bg_cache_fbo = 0;
    int    bg_cache_w = 0, bg_cache_h = 0;
    bool   bg_cache_valid = false;

    int   cached_grid_w = 0, cached_grid_h = 0;
    float cached_overview_z = 0.0f;
    float cached_total_w    = 0.0f, cached_total_h = 0.0f;
    float cached_cell_w     = 0.0f, cached_cell_h = 0.0f;
    float cached_aspect     = 1.0f;
    float cached_ox         = 0.0f, cached_oy = 0.0f;

    std::vector<glm::mat4> cached_models;
    bool models_dirty = true;

    /* OPT K: cached inverse matrices for raycasting */
    glm::mat4 cached_inv_proj;
    glm::mat4 cached_inv_view_scale;
    bool      raycast_cache_dirty = true;

void ensure_postprocess_fbos(int w, int h)
    {
        if (pp_width == w && pp_height == h && pp_scene_tex) return;

        auto setup_fbo = [](GLuint &tex, GLuint &fbo, int w, int h) {
            if (!tex) { GL_CALL(glGenTextures(1, &tex)); }
            if (!fbo) { GL_CALL(glGenFramebuffers(1, &fbo)); }
            GL_CALL(glBindTexture(GL_TEXTURE_2D, tex));
            GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
            GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, fbo));
            GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                           GL_TEXTURE_2D, tex, 0));
        };

        setup_fbo(pp_scene_tex, pp_scene_fbo, w, h);

        /* Attach depth renderbuffer to the scene FBO */
        if (!pp_scene_depth_rbo) {
            GL_CALL(glGenRenderbuffers(1, &pp_scene_depth_rbo));
        }
        GL_CALL(glBindRenderbuffer(GL_RENDERBUFFER, pp_scene_depth_rbo));
        GL_CALL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, w, h));
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_scene_fbo));
        GL_CALL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                          GL_RENDERBUFFER, pp_scene_depth_rbo));

        setup_fbo(pp_godrays_tex, pp_godrays_fbo, w, h);

        int bw = w / 2, bh = h / 2;
        setup_fbo(pp_bloom_tex[0], pp_bloom_fbo[0], bw, bh);
        setup_fbo(pp_bloom_tex[1], pp_bloom_fbo[1], bw, bh);

        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        pp_width = w;
        pp_height = h;
         setup_fbo(pp_occlude_tex, pp_occlude_fbo, w, h);
    }

    void update_cached_grid_metrics()
    {
        auto g = output->wset()->get_workspace_grid_size();
        auto og = output->get_layout_geometry();
        float aspect = (float)og.width / og.height;

        if (g.width == cached_grid_w && g.height == cached_grid_h
            && std::abs(cached_aspect - aspect) < 0.001f) return;

        cached_grid_w  = g.width;
        cached_grid_h  = g.height;
        cached_aspect  = aspect;

        float panel_w  = PANEL_SCALE * aspect;
        float panel_h  = PANEL_SCALE;
        cached_cell_w  = panel_w + GRID_GAP;
        cached_cell_h  = panel_h + GRID_GAP;
        cached_total_w = g.width  * cached_cell_w;
        cached_total_h = g.height * cached_cell_h;
        cached_ox      = -cached_total_w * 0.5f + cached_cell_w * 0.5f;
        cached_oy      =  cached_total_h * 0.5f - cached_cell_h * 0.5f;

        float fy = animation.projection[1][1];
        float fx = animation.projection[0][0];
        float dh = fy * (cached_total_h * 0.5f);
        float dw = fx * (cached_total_w * 0.5f);
        cached_overview_z = std::max(dh, dw) * 1.00f;
        models_dirty = true;
    }

    void generate_rounded_box_geometry()
    {
        float hw = 0.5f, hh = 0.5f;
        float D = BOX_DEPTH;
        float r = std::min(CORNER_RADIUS, std::min(hw, hh) * 0.9f);
        int segs = CORNER_SEGMENTS;

        /* ─── Build perimeter points + outward normals ──────────── */
        struct PP { float x, y, nx, ny; };
        std::vector<PP> perim;

        struct Corner { float cx, cy, start; };
        Corner corners[4] = {
            { hw - r,  hh - r, 0.0f },
            {-hw + r,  hh - r, (float)M_PI * 0.5f },
            {-hw + r, -hh + r, (float)M_PI },
            { hw - r, -hh + r, (float)M_PI * 1.5f },
        };

        for (int c = 0; c < 4; c++) {
            for (int s = 0; s <= segs; s++) {
                float t = (float)s / segs;
                float angle = corners[c].start + t * (float)M_PI * 0.5f;
                float nx = std::cos(angle);
                float ny = std::sin(angle);
                perim.push_back({
                    corners[c].cx + nx * r,
                    corners[c].cy + ny * r,
                    nx, ny
                });
            }
        }

        int N = (int)perim.size();

        /* ═══════════════════════════════════════════════════════════
         *  FRONT FACE: rounded rect with UVs (pos2 + uv2)
         *  Triangle fan from center
         * ═══════════════════════════════════════════════════════════*/
        std::vector<GLfloat> fv;
        std::vector<GLuint> fi;

        /* Center vertex */
        fv.push_back(0.0f); fv.push_back(0.0f);   /* pos */
        fv.push_back(0.5f); fv.push_back(0.5f);   /* uv  */

        for (int i = 0; i < N; i++) {
            fv.push_back(perim[i].x);
            fv.push_back(perim[i].y);
            fv.push_back(perim[i].x / (2.0f * hw) + 0.5f);
            fv.push_back(perim[i].y / (2.0f * hh) + 0.5f);
        }

        for (int i = 0; i < N; i++) {
            fi.push_back(0);
            fi.push_back(1 + i);
            fi.push_back(1 + (i + 1) % N);
        }

        front_index_count = (int)fi.size();

        if (panel_vbo) { GL_CALL(glDeleteBuffers(1, &panel_vbo)); panel_vbo = 0; }
        if (panel_ibo) { GL_CALL(glDeleteBuffers(1, &panel_ibo)); panel_ibo = 0; }

        GL_CALL(glGenBuffers(1, &panel_vbo));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, panel_vbo));
        GL_CALL(glBufferData(GL_ARRAY_BUFFER,
            fv.size() * sizeof(GLfloat), fv.data(), GL_STATIC_DRAW));

        GL_CALL(glGenBuffers(1, &panel_ibo));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_ibo));
        GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            fi.size() * sizeof(GLuint), fi.data(), GL_STATIC_DRAW));

        /* ═══════════════════════════════════════════════════════════
         *  SIDE STRIP + BACK FACE (pos3 + normal3)
         *
         *  Side: front perimeter → back perimeter with outward normals
         *  Front bevel: curved transition from front face to side
         *  Back bevel:  curved transition from side to back face
         *  Back face:   flat rounded rect at z = -D
         * ═══════════════════════════════════════════════════════════*/
        std::vector<GLfloat> sv;
        std::vector<GLuint> si;

        int bevel_segs = 4;
        float bevel_r = D * 0.4f;  /* how much of the depth is curved */
        float flat_depth = D - bevel_r * 2.0f;
        if (flat_depth < 0.0f) {
            bevel_r = D * 0.5f;
            flat_depth = 0.0f;
        }

        /* For each perimeter point, generate a profile curve:
         *   front face edge (z=0) → bevel out → straight side → bevel in → back face (z=-D)
         *
         *   Profile goes:
         *     Ring 0: z=0,          normal = (0,0,1) blended with outward
         *     Ring 1..bevel_segs:   front bevel curve
         *     Ring bevel_segs+1:    straight side start (z = -bevel_r)
         *     Ring bevel_segs+2:    straight side end   (z = -(D-bevel_r))
         *     Ring bevel_segs+3..+3+bevel_segs: back bevel curve
         *     Ring last:            z=-D, normal = (0,0,-1) blended with outward
         */

        int rings = 2 + bevel_segs * 2 + 2;  /* total rings in the profile */

        struct Ring { float z_off; float nx_blend; float nz; };
        std::vector<Ring> profile;

        /* Front bevel */
        for (int s = 0; s <= bevel_segs; s++) {
            float t = (float)s / bevel_segs;
            float angle = (float)M_PI * 0.5f * (1.0f - t);  /* 90° → 0° */
            float bx = std::cos(angle) * bevel_r;
            float bz = (1.0f - std::sin(angle)) * bevel_r;
            /* Normal: blend between forward (0,0,1) and outward */
            float n_out = std::cos(angle - (float)M_PI * 0.5f);
            float n_fwd = std::sin(angle - (float)M_PI * 0.5f);
            /* Actually compute from the curve tangent */
            float tnx = std::sin(angle);   /* outward component */
            float tnz = std::cos(angle);   /* forward component */
            profile.push_back({-bz, tnx, tnz});
        }

        /* Straight side */
        profile.push_back({-bevel_r, 1.0f, 0.0f});
        if (flat_depth > 0.001f) {
            profile.push_back({-(bevel_r + flat_depth), 1.0f, 0.0f});
        }

        /* Back bevel */
        for (int s = 0; s <= bevel_segs; s++) {
            float t = (float)s / bevel_segs;
            float angle = (float)M_PI * 0.5f * t;  /* 0° → 90° */
            float bz = bevel_r + flat_depth +
                       (1.0f - std::cos(angle)) * bevel_r;
            float tnx = std::cos(angle);
            float tnz = -std::sin(angle);  /* pointing backward */
            profile.push_back({-bz, tnx, tnz});
        }

        int num_rings = (int)profile.size();

        /* Generate vertices: N perimeter points × num_rings rings */
        for (int ring = 0; ring < num_rings; ring++) {
            auto& pr = profile[ring];
            for (int i = 0; i < N; i++) {
                /* Position: perimeter XY + profile Z offset */
                /* For bevel, push outward slightly based on profile */
                float bevel_expand = 0.0f;
                if (ring < (int)profile.size()) {
                    /* The bevel curve pushes geometry inward from the edge
                       to create the rounded profile */
                    if (ring <= bevel_segs) {
                        float t = (float)ring / bevel_segs;
                        float angle = (float)M_PI * 0.5f * (1.0f - t);
                        bevel_expand = -(bevel_r - std::cos(angle) * bevel_r);
                    } else if (ring >= num_rings - bevel_segs - 1) {
                        int br = ring - (num_rings - bevel_segs - 1);
                        float t = (float)br / bevel_segs;
                        float angle = (float)M_PI * 0.5f * t;
                        bevel_expand = -(bevel_r - std::cos(angle) * bevel_r);
                    }
                }

                float px = perim[i].x + perim[i].nx * bevel_expand;
                float py = perim[i].y + perim[i].ny * bevel_expand;
                float pz = pr.z_off;

                /* Normal: combine perimeter outward with profile Z component */
                float fnx = perim[i].nx * pr.nx_blend;
                float fny = perim[i].ny * pr.nx_blend;
                float fnz = pr.nz;
                float nlen = std::sqrt(fnx*fnx + fny*fny + fnz*fnz);
                if (nlen > 0.001f) { fnx /= nlen; fny /= nlen; fnz /= nlen; }

                sv.push_back(px);  sv.push_back(py);  sv.push_back(pz);
                sv.push_back(fnx); sv.push_back(fny); sv.push_back(fnz);
            }
        }

        /* Side strip indices: connect adjacent rings */
        for (int ring = 0; ring < num_rings - 1; ring++) {
            for (int i = 0; i < N; i++) {
                int next = (i + 1) % N;
                int r0i = ring * N + i;
                int r0n = ring * N + next;
                int r1i = (ring + 1) * N + i;
                int r1n = (ring + 1) * N + next;

                si.push_back(r0i); si.push_back(r1i); si.push_back(r0n);
                si.push_back(r0n); si.push_back(r1i); si.push_back(r1n);
            }
        }

        /* Back face: triangle fan at z = -D with normal (0,0,-1) */
        int back_center = (int)(sv.size() / 6);
        sv.push_back(0.0f); sv.push_back(0.0f); sv.push_back(-D);
        sv.push_back(0.0f); sv.push_back(0.0f); sv.push_back(-1.0f);

        /* Back perimeter vertices (inset by bevel) */
        float back_inset = bevel_r;  /* back bevel pulls edge inward */
        for (int i = 0; i < N; i++) {
            float px = perim[i].x - perim[i].nx * back_inset;
            float py = perim[i].y - perim[i].ny * back_inset;
            sv.push_back(px);   sv.push_back(py);   sv.push_back(-D);
            sv.push_back(0.0f); sv.push_back(0.0f); sv.push_back(-1.0f);
        }

        /* Back face triangles (reversed winding) */
        for (int i = 0; i < N; i++) {
            si.push_back(back_center);
            si.push_back(back_center + 1 + (i + 1) % N);
            si.push_back(back_center + 1 + i);
        }

        sides_index_count = (int)si.size();

        if (box_sides_vbo) { GL_CALL(glDeleteBuffers(1, &box_sides_vbo)); box_sides_vbo = 0; }
        if (box_sides_ibo) { GL_CALL(glDeleteBuffers(1, &box_sides_ibo)); box_sides_ibo = 0; }

        GL_CALL(glGenBuffers(1, &box_sides_vbo));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, box_sides_vbo));
        GL_CALL(glBufferData(GL_ARRAY_BUFFER,
            sv.size() * sizeof(GLfloat), sv.data(), GL_STATIC_DRAW));

        GL_CALL(glGenBuffers(1, &box_sides_ibo));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_sides_ibo));
        GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            si.size() * sizeof(GLuint), si.data(), GL_STATIC_DRAW));

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    void rebuild_model_cache(int gw, int gh)
    {
        if (!models_dirty &&
            (int)cached_models.size() == gw*gh) return;
        cached_models.resize(gw * gh);
        auto cws = output->wset()->get_current_workspace();
        auto og  = output->get_layout_geometry();
        float aspect = (float)og.width / og.height;
        for (int wy = 0; wy < gh; wy++) {
            for (int wx = 0; wx < gw; wx++) {
                int rwx = (cws.x + wx) % gw;
                float px = cached_ox + rwx * cached_cell_w;
                float py = -(cached_oy - (float)wy * cached_cell_h);
                auto T = glm::translate(glm::mat4(1), glm::vec3(px, py, 0.0f));
                auto S = glm::scale(glm::mat4(1),
                    glm::vec3(PANEL_SCALE * aspect, PANEL_SCALE, 1.0f));
                cached_models[wy*gw + wx] = T * S;
            }
        }
        models_dirty = false;
    }

    float identity_z_offset;
    wf_cube_animation_attribs animation;

    wf::option_wrapper_t<double> XVelocity{"vertical_expo/speed_spin_horiz"},
        YVelocity{"vertical_expo/speed_spin_vert"},
        ZVelocity{"vertical_expo/speed_zoom"};
    wf::option_wrapper_t<double> zoom_opt{"vertical_expo/zoom"};
    wf::option_wrapper_t<bool>   tron{"vertical_expo/tron"};
    wf::option_wrapper_t<bool>   star_background{"vertical_expo/star_background"};
    wf::option_wrapper_t<bool>   use_light{"vertical_expo/light"};
    wf::option_wrapper_t<int>    use_deform{"vertical_expo/deform"};

    std::string last_background_mode;
    std::unique_ptr<wf_cube_background_base> background;
    wf::option_wrapper_t<std::string> background_mode{"vertical_expo/background_mode"};
    bool tessellation_support = false;

    std::unique_ptr<wf::input_grab_t> input_grab;
    wf::plugin_activation_data_t grab_interface{
        .name = "cube",
        .capabilities = wf::CAPABILITY_MANAGE_COMPOSITOR,
        .cancel = [=]() { deactivate(); },
    };

    /* ═════════════════════════════════════════════════════════════════════
     *  RENDER NODE
     * ═════════════════════════════════════════════════════════════════════*/

    class cube_render_node_t : public wf::scene::node_t
    {
        class windows_only_workspace_node_t : public wf::scene::node_t {
            wf::output_t *output; wf::point_t workspace;
        public:
            windows_only_workspace_node_t(wf::output_t *o, wf::point_t ws)
                : node_t(false), output(o), workspace(ws) {}
            void gen_render_instances(
                std::vector<wf::scene::render_instance_uptr>& inst,
                wf::scene::damage_callback cb, wf::output_t *on) override
            {
                if (on != output) return;
                auto views = output->wset()->get_views();
                auto og = output->get_layout_geometry();
                wf::geometry_t wg{og.x+workspace.x*og.width,
                    og.y+workspace.y*og.height, og.width, og.height};
                for (auto& v : views) {
                    if (!v->is_mapped()) continue;
                    auto vws = output->wset()->get_view_main_workspace(v);
                    auto vg  = v->get_geometry();
                    bool ol  = !(vg.x+vg.width<=wg.x || vg.x>=wg.x+wg.width ||
                                 vg.y+vg.height<=wg.y || vg.y>=wg.y+wg.height);
                    if (vws != workspace && !ol) continue;
                    auto n = v->get_root_node();
                    if (n) n->gen_render_instances(inst, cb, on);
                }
            }
            wf::geometry_t get_bounding_box() override
            { return output->get_layout_geometry(); }
        };

        class desktop_only_workspace_node_t : public wf::scene::node_t {
            wf::output_t *output; wf::point_t workspace;
        public:
            desktop_only_workspace_node_t(wf::output_t *o, wf::point_t ws)
                : node_t(false), output(o), workspace(ws) {}
            void gen_render_instances(
                std::vector<wf::scene::render_instance_uptr>& inst,
                wf::scene::damage_callback cb, wf::output_t *on) override
            {
                if (on != output) return;
                auto r = output->node_for_layer(wf::scene::layer::BACKGROUND);
                if (r) r->gen_render_instances(inst, cb, on);
                auto b = output->node_for_layer(wf::scene::layer::BOTTOM);
                if (b) b->gen_render_instances(inst, cb, on);
            }
            wf::geometry_t get_bounding_box() override
            { return output->get_layout_geometry(); }
        };

    public:
        struct WsRenderState {
            std::vector<wf::scene::render_instance_uptr> inst;
            wf::region_t            dmg;
            wf::auxilliary_buffer_t fb;
            bool allocated = false;
            int drag_damage_frames = 0;
            bool desktop_rendered = false;

            std::unique_ptr<wf::scene::render_instance_manager_t> win_mgr;
            wf::region_t dmg_win;

            std::vector<wf::scene::render_instance_uptr> win_inst_cache;
            bool win_inst_dirty = true;

            float lod_scale = 1.0f;
        };

    private:
        class cube_render_instance_t : public wf::scene::render_instance_t
        {
            std::shared_ptr<cube_render_node_t> self;
            wf::scene::damage_callback push_damage;

            std::vector<WsRenderState> ws_states;
            int grid_w = 0, grid_h = 0;

            std::vector<wf::scene::render_instance_uptr> panel_inst_cache;
            bool panels_generated_this_frame = false;

            wf::signal::connection_t<wf::scene::node_damage_signal> on_dmg =
                [=](wf::scene::node_damage_signal *e)
                { push_damage(e->region); };

            void ensure_fbo(wf::auxilliary_buffer_t& fb,
                            bool& allocated,
                            wf::region_t& dmg,
                            wf::dimensions_t dim,
                            float scale,
                            const wf::geometry_t& full_geom)
            {
                if (allocated) return;
                fb.allocate(dim, scale);
                allocated = true;
                dmg |= full_geom;
            }

            static float compute_lod_scale(float proj_px, float screen_px)
            {
                float ratio = proj_px / screen_px;
                float s = std::clamp(ratio * 1.5f, LOD_MIN_SCALE, LOD_MAX_SCALE);
                return std::round(s * 4.0f) / 4.0f;
            }

        public:
            cube_render_instance_t(cube_render_node_t *s,
                                   wf::scene::damage_callback pd)
            {
                self = std::dynamic_pointer_cast<cube_render_node_t>(
                    s->shared_from_this());
                push_damage = pd;
                s->connect(&on_dmg);

                grid_w = (int)s->ws.size();
                grid_h = 1 + (int)s->ws_all_rows.size();
                ws_states.resize(grid_w * grid_h);

                for (int i = 0; i < grid_w; i++) {
                    auto& st = ws_states[i];
                    auto cb = [=](const wf::region_t& d) {
                        ws_states[i].dmg |= d;
                        pd(s->get_bounding_box());
                    };
                    s->ws[i]->gen_render_instances(st.inst, cb, s->cube->output);
                    st.dmg |= s->ws[i]->get_bounding_box();
                }

                for (int i = 0; i < grid_w; i++) {
                    auto& st = ws_states[i];
                    auto cb = [=](const wf::region_t& d) {
                        ws_states[i].dmg_win |= d;
                        ws_states[i].dmg     |= d;
                        ws_states[i].win_inst_dirty = true;
                        push_damage(self->get_bounding_box());
                    };
                    std::vector<wf::scene::node_ptr> n;
                    n.push_back(s->ws_win[i]);
                    st.win_mgr =
                        std::make_unique<wf::scene::render_instance_manager_t>(
                            n, cb, s->cube->output);
                    const int B = 1e5;
                    st.win_mgr->set_visibility_region(
                        wf::geometry_t{-B,-B,2*B,2*B});
                    st.dmg     |= s->ws_win[i]->get_bounding_box();
                    st.dmg_win |= s->ws_win[i]->get_bounding_box();
                    st.win_inst_dirty = true;
                }

                for (int r = 0; r < (int)s->ws_all_rows.size(); r++) {
                    for (int i = 0; i < grid_w; i++) {
                        int idx = (r+1)*grid_w + i;
                        auto& st = ws_states[idx];

                        auto cb = [=](const wf::region_t& d) {
                            ws_states[idx].dmg |= d;
                            pd(s->get_bounding_box());
                        };
                        s->ws_all_rows[r][i]->gen_render_instances(
                            st.inst, cb, s->cube->output);
                        st.dmg |= s->ws_all_rows[r][i]->get_bounding_box();

                        auto cbw = [=](const wf::region_t& d) {
                            ws_states[idx].dmg_win |= d;
                            ws_states[idx].dmg     |= d;
                            ws_states[idx].win_inst_dirty = true;
                            push_damage(self->get_bounding_box());
                        };
                        std::vector<wf::scene::node_ptr> nw;
                        nw.push_back(s->ws_win_rows[r][i]);
                        st.win_mgr =
                            std::make_unique<wf::scene::render_instance_manager_t>(
                                nw, cbw, s->cube->output);
                        const int B = 1e5;
                        st.win_mgr->set_visibility_region(
                            wf::geometry_t{-B,-B,2*B,2*B});
                        st.dmg     |=
                            s->ws_win_rows[r][i]->get_bounding_box();
                        st.dmg_win |=
                            s->ws_win_rows[r][i]->get_bounding_box();
                        st.win_inst_dirty = true;
                    }
                }
            }

            ~cube_render_instance_t()
            { for (auto& st : ws_states) st.fb.free(); }

            void schedule_instructions(
                std::vector<wf::scene::render_instruction_t>& instr,
                const wf::render_target_t& tgt, wf::region_t& dmg) override
            {
                auto bb = self->get_bounding_box();
                dmg ^= bb;
                float sc = self->cube->output->handle->scale;
                auto  og = self->cube->output->get_layout_geometry();
                wf::dimensions_t full_dim{og.width, og.height};
                float ow = (float)og.width, oh = (float)og.height;

                instr.push_back({
                    .instance = this,
                    .target   = tgt.translated(-wf::origin(bb)),
                    .damage   = dmg & bb,
                });

                panels_generated_this_frame = false;

                /* Drag: dirty overlapping workspaces */
                if (self->drag_active) {
                    auto grid = self->cube->output->wset()
                                    ->get_workspace_grid_size();
                    auto& wg = self->drag_window_geom;
                    auto& pg = self->drag_window_geom_prev;
                    for (int wy = 0; wy < grid.height; wy++) {
                        for (int wx = 0; wx < grid.width; wx++) {
                            wf::geometry_t cell = {
                                wx*og.width, wy*og.height,
                                og.width, og.height};
                            auto overlaps = [&](const wf::geometry_t& g) {
                                return g.x<cell.x+cell.width  &&
                                       g.x+g.width>cell.x    &&
                                       g.y<cell.y+cell.height &&
                                       g.y+g.height>cell.y;
                            };
                            if (!overlaps(wg) && !overlaps(pg)) continue;
                            int idx = wy*grid_w + wx;
                            if (idx < (int)ws_states.size()) {
                                ws_states[idx].dmg |= bb;
                                ws_states[idx].win_inst_dirty = true;
                                ws_states[idx].drag_damage_frames = 3;
                            }
                        }
                    }
                }

                for (int idx = 0; idx < (int)ws_states.size(); idx++) {
                    if (ws_states[idx].drag_damage_frames > 0) {
                        ws_states[idx].dmg |= bb;
                        ws_states[idx].win_inst_dirty = true;
                        ws_states[idx].drag_damage_frames--;
                    }
                }

                auto& cube = self->cube;
                cube->update_cached_grid_metrics();
                cube->rebuild_model_cache(grid_w, grid_h);

                float z   = cube->animation.cube_animation.zoom;
                auto sm   = glm::scale(glm::mat4(1), glm::vec3(1.0f/z));
                auto vp_approx = cube->animation.projection *
                                 cube->animation.view * sm;

                int fbo_renders_this_frame = 0;

                for (int wy = 0; wy < grid_h; wy++) {
                    for (int wx = 0; wx < grid_w; wx++) {
                        int idx = wy*grid_w + wx;
                        auto& st = ws_states[idx];

                        float desired_lod = LOD_MAX_SCALE;
                        bool zooming = cube->zoom_anim.running;
                        if (!zooming) {
                            desired_lod = LOD_MAX_SCALE;
                            if (idx < (int)cube->cached_models.size()) {
                                glm::mat4 mvp = vp_approx * cube->cached_models[idx];
                                float proj_px = projected_quad_size_px(mvp, ow, oh);
                                desired_lod = compute_lod_scale(proj_px,
                                    std::max(ow, oh));
                            }

                            if (st.allocated &&
                                std::abs(st.lod_scale - desired_lod) > LOD_HYSTERESIS)
                            {
                                st.fb.free();
                                st.allocated = false;
                                st.desktop_rendered = false;
                                st.dmg |= og;
                            }
                            st.lod_scale = desired_lod;
                        }

                        float effective_scale = sc * desired_lod;
                        ensure_fbo(st.fb, st.allocated, st.dmg,
                                   full_dim, effective_scale, og);

                        if (st.dmg.empty()) continue;

                        if (!self->drag_active &&
                            fbo_renders_this_frame >= MAX_FBO_RENDERS_PER_FRAME) {
                            continue;
                        }
                        fbo_renders_this_frame++;

                        /* Pass A: desktop */
                        {
                            auto b = (wy==0)
                                ? self->ws[wx]->get_bounding_box()
                                : self->ws_all_rows[wy-1][wx]->get_bounding_box();
                            wf::render_target_t t{st.fb};
                            t.geometry = b;
                            t.scale    = effective_scale;
                            wf::render_pass_params_t p;
                            p.instances        = &st.inst;
                            p.damage           = st.dmg;
                            p.reference_output = self->cube->output;
                            p.target           = t;
                            p.flags = wf::RPASS_CLEAR_BACKGROUND |
                                      wf::RPASS_EMIT_SIGNALS;
                            wf::render_pass_t::run(p);
                        }

                        /* Pass B: windows + panels */
                        {
                            auto cws = self->cube->output->wset()
                                           ->get_current_workspace();
                            auto grid = self->cube->output->wset()
                                            ->get_workspace_grid_size();
                            int tx_ = (cws.x + wx) % grid.width;
                            int ty_ = (cws.y + wy) % grid.height;

                            wf::render_target_t ft{st.fb};
                            ft.geometry = wf::geometry_t{
                                og.x + tx_*og.width,
                                og.y + ty_*og.height,
                                og.width, og.height};
                            ft.scale = effective_scale;

                            if (st.win_inst_dirty) {
                                st.win_inst_cache.clear();
                                auto dd = [](wf::region_t){};
                                if (wy==0)
                                    self->regenerate_workspace_instances(
                                        wx, st.win_inst_cache, dd);
                                else
                                    self->regenerate_workspace_instances_row(
                                        wy-1, wx, st.win_inst_cache, dd);
                                st.win_inst_dirty = false;
                            }

                            wf::region_t full_r = ft.geometry;
                            wf::render_pass_params_t p;
                            p.instances        = &st.win_inst_cache;
                            p.damage           = full_r;
                            p.reference_output = self->cube->output;
                            p.target           = ft;
                            p.flags = wf::RPASS_EMIT_SIGNALS;
                            wf::render_pass_t::run(p);

                            if (!panels_generated_this_frame) {
                                panels_generated_this_frame = true;
                                panel_inst_cache.clear();
                                auto dd2 = [](wf::region_t){};
                                auto top = self->cube->output->node_for_layer(
                                    wf::scene::layer::TOP);
                                if (top) top->gen_render_instances(
                                    panel_inst_cache, dd2, self->cube->output);
                                auto ovr = self->cube->output->node_for_layer(
                                    wf::scene::layer::OVERLAY);
                                if (ovr) ovr->gen_render_instances(
                                    panel_inst_cache, dd2, self->cube->output);
                            }
                            if (!panel_inst_cache.empty()) {
                                wf::render_target_t pt{st.fb};
                                pt.geometry = wf::geometry_t{
                                    og.x, og.y, og.width, og.height};
                                pt.scale = effective_scale;
                                wf::region_t pr = pt.geometry;
                                wf::render_pass_params_t pp;
                                pp.instances        = &panel_inst_cache;
                                pp.damage           = pr;
                                pp.reference_output = self->cube->output;
                                pp.target           = pt;
                                pp.flags = wf::RPASS_EMIT_SIGNALS;
                                wf::render_pass_t::run(pp);
                            }
                        }

                        st.dmg.clear();
                        st.dmg_win.clear();
                    }
                }

                if (fbo_renders_this_frame >= MAX_FBO_RENDERS_PER_FRAME)
                    self->cube->output->render->schedule_redraw();
            }

            void render(const wf::scene::render_instruction_t& d) override
            { self->cube->render_from_states(d, ws_states, grid_w, grid_h); }

            void compute_visibility(wf::output_t *o,
                                    wf::region_t& vis) override
            {
                for (int i = 0; i < grid_w; i++) {
                    auto& st = ws_states[i];
                    wf::region_t r = self->ws[i]->get_bounding_box();
                    for (auto& c : st.inst) c->compute_visibility(o, r);
                    if (st.win_mgr)
                        for (auto& c : st.win_mgr->get_instances())
                            c->compute_visibility(o, r);
                }
                for (int r = 0; r < (int)self->ws_all_rows.size(); r++) {
                    for (int i = 0; i < grid_w; i++) {
                        int idx = (r+1)*grid_w + i;
                        auto& st = ws_states[idx];
                        wf::region_t rr =
                            self->ws_all_rows[r][i]->get_bounding_box();
                        for (auto& c : st.inst) c->compute_visibility(o, rr);
                        if (st.win_mgr)
                            for (auto& c : st.win_mgr->get_instances())
                                c->compute_visibility(o, rr);
                    }
                }
            }
        }; /* cube_render_instance_t */

    public:
        cube_render_node_t(wayfire_cube *cube) : node_t(false)
        {
            this->cube = cube;
            auto g = cube->output->wset()->get_workspace_grid_size();
            auto y = cube->output->wset()->get_current_workspace().y;
            for (int i = 0; i < g.width; i++) {
                ws.push_back(
                    std::make_shared<desktop_only_workspace_node_t>(
                        cube->output, wf::point_t{i, y}));
                ws_win.push_back(
                    std::make_shared<windows_only_workspace_node_t>(
                        cube->output, wf::point_t{i, y}));
            }
            for (int ro = 1; ro < g.height; ro++) {
                int ty = (y+ro) % g.height;
                std::vector<std::shared_ptr<wf::scene::node_t>> rd, rw;
                for (int i = 0; i < g.width; i++) {
                    rd.push_back(
                        std::make_shared<desktop_only_workspace_node_t>(
                            cube->output, wf::point_t{i, ty}));
                    rw.push_back(
                        std::make_shared<windows_only_workspace_node_t>(
                            cube->output, wf::point_t{i, ty}));
                }
                ws_all_rows.push_back(rd);
                ws_win_rows.push_back(rw);
            }
        }

        void gen_render_instances(
            std::vector<wf::scene::render_instance_uptr>& i,
            wf::scene::damage_callback pd, wf::output_t *on) override
        {
            if (on != cube->output) return;
            i.push_back(std::make_unique<cube_render_instance_t>(this, pd));
        }

        wf::geometry_t get_bounding_box()
        { return cube->output->get_layout_geometry(); }

        void damage_all_workspace_windows()
        {
            for (auto& n : ws_win)
                wf::scene::damage_node(n, n->get_bounding_box());
            for (auto& r : ws_win_rows)
                for (auto& n : r)
                    wf::scene::damage_node(n, n->get_bounding_box());
        }
        void damage_all_workspace_desktops()
        {
            for (auto& n : ws)
                wf::scene::damage_node(n, n->get_bounding_box());
            for (auto& r : ws_all_rows)
                for (auto& n : r)
                    wf::scene::damage_node(n, n->get_bounding_box());
        }
        void regenerate_workspace_instances(int idx,
            std::vector<wf::scene::render_instance_uptr>& i,
            wf::scene::damage_callback cb)
        {
            if (idx>=0 && idx<(int)ws_win.size())
                ws_win[idx]->gen_render_instances(i, cb, cube->output);
        }
        void regenerate_workspace_instances_row(int row, int idx,
            std::vector<wf::scene::render_instance_uptr>& i,
            wf::scene::damage_callback cb)
        {
            if (row>=0 && row<(int)ws_win_rows.size() &&
                idx >=0 && idx <(int)ws_win_rows[row].size())
                ws_win_rows[row][idx]->gen_render_instances(
                    i, cb, cube->output);
        }

        bool           drag_active           = false;
        wf::geometry_t drag_window_geom      = {0,0,0,0};
        wf::geometry_t drag_window_geom_prev = {0,0,0,0};

    private:
        std::vector<std::shared_ptr<wf::scene::node_t>> ws, ws_win;
        std::vector<std::vector<std::shared_ptr<wf::scene::node_t>>>
            ws_all_rows, ws_win_rows;
        wayfire_cube *cube;
    };

    std::shared_ptr<cube_render_node_t> render_node;

    /* ═════════════════════════════════════════════════════════════════════
     *  HELPERS
     * ═════════════════════════════════════════════════════════════════════*/

    int get_num_faces()
    { return output->wset()->get_workspace_grid_size().width; }

    void reload_background()
    {
        if (last_background_mode == (std::string)background_mode) return;
        last_background_mode = background_mode;
        if      (last_background_mode=="simple")
            background = std::make_unique<wf_cube_simple_background>();
        else if (last_background_mode=="skydome")
            background = std::make_unique<wf_cube_background_skydome>(output);
        else if (last_background_mode=="cubemap")
            background = std::make_unique<wf_cube_background_cubemap>();
        else
            background = std::make_unique<wf_cube_simple_background>();
    }

    float calculate_overview_z()
    { update_cached_grid_metrics(); return cached_overview_z; }

    glm::vec2 workspace_grid_center(int wx, int wy)
    {
        update_cached_grid_metrics();
        return { cached_ox + wx * cached_cell_w,
                -cached_oy + wy * cached_cell_h };
    }

    glm::mat4 calculate_model_matrix(int face_i, float v_offset=0.0f,
            float scale=1.0f, bool is_window=false, float depth=0.0f)
    {
        update_cached_grid_metrics();
        auto cws = output->wset()->get_current_workspace();
        int wx = (cws.x + face_i) % cached_grid_w;
        int wy = 0;
        if (std::abs(v_offset) > 1e-6f)
            wy = (int)std::round(-v_offset / cached_cell_h);
        float px = cached_ox + wx * cached_cell_w;
        float py = -cached_oy + wy * cached_cell_h;
        float pz = (is_window ? 0.05f : 0.0f) - depth*0.01f;
        auto T = glm::translate(glm::mat4(1), glm::vec3(px,py,pz));
        auto S = glm::scale(glm::mat4(1),
            glm::vec3(PANEL_SCALE*cached_aspect*scale,
                      PANEL_SCALE*scale, 1));
        return T * S;
    }

    glm::mat4 output_transform(const wf::render_target_t& t)
    {
        return wf::gles::render_target_gl_to_framebuffer(t) *
               glm::scale(glm::mat4(1), {1,-1,1});
    }

    glm::mat4 calculate_vp_matrix(const wf::render_target_t& dest)
    {
        float z = animation.cube_animation.zoom;
        auto sm = glm::scale(glm::mat4(1), glm::vec3(1.0f/z));
        return output_transform(dest) * animation.projection *
               animation.view * sm;
    }

    void update_view_matrix()
    {
        bool use_zoom = zoom_anim.running || animation.in_exit ||
                        zoom_state==ZoomState::ZOOMED_IN  ||
                        zoom_state==ZoomState::ZOOMING_IN ||
                        zoom_state==ZoomState::ZOOMING_OUT;
        float dist, lx, ly;
        if (use_zoom) {
            lx   = zoom_anim.current_pos.x;
            ly   = zoom_anim.current_pos.y;
            dist = zoom_anim.current_pos.z;
            animation.cube_animation.offset_z.set(dist,dist);
        } else {
            dist = animation.cube_animation.offset_z;
            lx   = zoom_anim.current_pos.x;
            ly   = zoom_anim.current_pos.y;
        }
        animation.view = glm::lookAt(
            glm::vec3(lx,ly,dist),
            glm::vec3(lx,ly,0.0f),
            glm::vec3(0,1,0));

        raycast_cache_dirty = true;
    }

    float calculate_close_z()
    {
        float f = animation.projection[1][1];
        return f * PANEL_SCALE * 0.5f;
    }

    void zoom_into_workspace(wf::point_t ws)
    {
        if (zoom_state==ZoomState::ZOOMING_IN ||
            zoom_state==ZoomState::ZOOMED_IN) return;
        zoom_target_ws=ws; selected_ws=ws;
        zoom_state=ZoomState::ZOOMING_IN;
        auto center=workspace_grid_center(ws.x,ws.y);
        zoom_anim.start(zoom_anim.current_pos,
            glm::vec3(center.x,center.y,calculate_close_z()),0.45f);
        output->render->schedule_redraw();
    }

    void zoom_out_to_overview()
    {
        if (zoom_state==ZoomState::ZOOMING_OUT ||
            zoom_state==ZoomState::OVERVIEW) return;
        zoom_state=ZoomState::ZOOMING_OUT;
        zoom_anim.start(zoom_anim.current_pos,
            glm::vec3(0.0f,0.0f,calculate_overview_z()),0.45f);
        output->render->schedule_redraw();
    }

    void update_zoom_state()
    {
        if      (zoom_state==ZoomState::ZOOMING_IN  && !zoom_anim.running)
            zoom_state=ZoomState::ZOOMED_IN;
        else if (zoom_state==ZoomState::ZOOMING_OUT && !zoom_anim.running)
            zoom_state=ZoomState::OVERVIEW;
    }

    bool is_zooming() const
    { return zoom_state==ZoomState::ZOOMING_IN ||
             zoom_state==ZoomState::ZOOMING_OUT; }
    bool is_zoomed_in() const
    { return zoom_state==ZoomState::ZOOMED_IN; }

    struct Ray     { glm::vec3 origin, dir; };
    struct HitInfo { wf::point_t ws={-1,-1}; float t=FLT_MAX;
                     glm::vec2 local_uv={0,0}; int row_offset=0; };

    void ensure_raycast_cache()
    {
        if (!raycast_cache_dirty) return;
        cached_inv_proj = glm::inverse(animation.projection);
        float z = animation.cube_animation.zoom;
        auto sm = glm::scale(glm::mat4(1), glm::vec3(1.0f/z));
        cached_inv_view_scale = glm::inverse(animation.view * sm);
        raycast_cache_dirty = false;
    }

    Ray screen_to_world_ray(float sx, float sy, wf::output_t *out)
    {
        ensure_raycast_cache();
        auto bb = out->get_layout_geometry();
        float nx = 2.0f*sx/bb.width  - 1.0f;
        float ny = 2.0f*sy/bb.height - 1.0f;
        glm::vec4 e4 = cached_inv_proj*glm::vec4(nx,ny,-1,1);
        e4/=e4.w;
        glm::vec4 c4=cached_inv_view_scale*glm::vec4(0,0,0,1); c4/=c4.w;
        auto d4 =cached_inv_view_scale*glm::vec4(glm::vec3(e4),0);
        return {glm::vec3(c4), glm::normalize(glm::vec3(d4))};
    }

    HitInfo raycast_grid(const glm::vec3& ro, const glm::vec3& rd,
                         wf::output_t *out, bool window_layer)
    {
        HitInfo hit;
        float pz = window_layer ? 0.05f : 0.0f;

        if (std::abs(rd.z) < 1e-6f) return hit;
        float t = (pz - ro.z) / rd.z;
        if (t < 0.0f) return hit;

        glm::vec3 p = ro + t * rd;

        update_cached_grid_metrics();
        auto grid = out->wset()->get_workspace_grid_size();
        auto cws  = out->wset()->get_current_workspace();

        float cy_base = -cached_oy;

        float fy = (p.y - cy_base) / cached_cell_h;
        float fx = (p.x - cached_ox) / cached_cell_w;

        int wx = std::clamp((int)std::round(fx), 0, grid.width  - 1);
        int wy = std::clamp((int)std::round(fy), 0, grid.height - 1);

        float cell_cx = cached_ox  + wx * cached_cell_w;
        float cell_cy = cy_base    + wy * cached_cell_h;

        float panel_hw = PANEL_SCALE * cached_aspect * 0.5f;
        float panel_hh = PANEL_SCALE * 0.5f;
        float lx = (p.x - cell_cx) / (panel_hw * 2.0f);
        float ly = (p.y - cell_cy) / (panel_hh * 2.0f);

        float uv_x = std::clamp(lx + 0.5f, 0.0f, 1.0f);
        float uv_y = std::clamp(ly + 0.5f, 0.0f, 1.0f);

        if (!is_dragging_window) {
            if (std::abs(lx) > 0.55f || std::abs(ly) > 0.55f)
                return hit;
        }

        hit.ws = {(cws.x + wx) % grid.width,
                  (cws.y + wy) % grid.height};
        hit.t  = t;
        hit.local_uv = {uv_x, uv_y};
        hit.row_offset = wy;
        return hit;
    }

    HitInfo raycast_to_workspace(const glm::vec3& o, const glm::vec3& d,
                                 wf::output_t *out)
    { return raycast_grid(o,d,out,false); }
    HitInfo raycast_at_window_depth(const glm::vec3& o, const glm::vec3& d,
                                    wf::output_t *out)
    { return raycast_grid(o,d,out,true); }

    wayfire_toplevel_view find_window_at_cursor_on_face(
        wf::pointf_t vc, const wf::point_t&)
    {
        for (auto& v : output->wset()->get_views()) {
            if (!v->is_mapped()) continue;
            auto tv=wf::toplevel_cast(v); if (!tv) continue;
            auto g=tv->get_geometry();
            if (vc.x>=g.x&&vc.x<=g.x+g.width&&
                vc.y>=g.y&&vc.y<=g.y+g.height) return tv;
        }
        return nullptr;
    }

    /* ═════════════════════════════════════════════════════════════════════
     *  GL LOAD — includes 3D box geometry
     * ═════════════════════════════════════════════════════════════════════*/
    void load_program()
    {
        tessellation_support = false;
        godrays_program.set_simple(
            OpenGL::compile_program(godrays_vs, godrays_fs));
        bloom_extract_program.set_simple(
            OpenGL::compile_program(godrays_vs, bloom_extract_fs));
        blur_program.set_simple(
            OpenGL::compile_program(godrays_vs, blur_fs));
        composite_program.set_simple(
            OpenGL::compile_program(godrays_vs, composite_fs));
        orb_program.set_simple(
            OpenGL::compile_program(orb_vs, orb_fs));
        neon_edge_program.set_simple(
            OpenGL::compile_program(neon_edge_vs, neon_edge_fs));
        program.set_simple(OpenGL::compile_program(panel_vs, panel_fs));
        cap_program.set_simple(OpenGL::compile_program(panel_vs, panel_fs));
        background_program.set_simple(OpenGL::compile_program(bg_vs, bg_fs));
        cursor_program.set_simple(OpenGL::compile_program(ring_vs, ring_fs));
        beam_program.set_simple(OpenGL::compile_program(ring_vs, ring_fs));
        blit_program.set_simple(OpenGL::compile_program(blit_vs, blit_fs));
        silhouette_program.set_simple(
            OpenGL::compile_program(silhouette_vs, silhouette_fs));
        light_disc_program.set_simple(
            OpenGL::compile_program(light_disc_vs, light_disc_fs));
        box_color_program.set_simple(
            OpenGL::compile_program(box_color_vs, box_color_fs));

        if (!cursor_vbo) {
            const int S=48;
            std::vector<GLfloat> cv; cv.reserve(2*(S+2));
            cv.push_back(0); cv.push_back(0);
            for (int i=0;i<=S;i++) {
                float a=(float)i/S*2.0f*(float)M_PI;
                cv.push_back(std::cos(a)); cv.push_back(std::sin(a));
            }
            GL_CALL(glGenBuffers(1,&cursor_vbo));
            GL_CALL(glBindBuffer(GL_ARRAY_BUFFER,cursor_vbo));
            GL_CALL(glBufferData(GL_ARRAY_BUFFER,
                cv.size()*sizeof(GLfloat),cv.data(),GL_STATIC_DRAW));
        }
        if (!background_vbo) {
            static const GLfloat q[]={-1,-1,1,-1,-1,1,1,1};
            GL_CALL(glGenBuffers(1,&background_vbo));
            GL_CALL(glBindBuffer(GL_ARRAY_BUFFER,background_vbo));
            GL_CALL(glBufferData(GL_ARRAY_BUFFER,sizeof(q),q,GL_STATIC_DRAW));
        }

        /* Front face quad VBO (interleaved pos2 + uv2) */
        if (!panel_vbo) {
            static const GLfloat data[] = {
                -0.5f, 0.5f,  0, 1,
                 0.5f, 0.5f,  1, 1,
                 0.5f,-0.5f,  1, 0,
                -0.5f,-0.5f,  0, 0,
            };
            GL_CALL(glGenBuffers(1,&panel_vbo));
            GL_CALL(glBindBuffer(GL_ARRAY_BUFFER,panel_vbo));
            GL_CALL(glBufferData(GL_ARRAY_BUFFER,
                sizeof(data),data,GL_STATIC_DRAW));
        }

  generate_rounded_box_geometry();
  
        /* Box sides IBO: 5 faces × 6 indices = 30 */
        if (!box_sides_ibo) {
            static const GLuint idx[] = {
                /* back   */  0, 1, 2,   0, 2, 3,
                /* left   */  4, 5, 6,   4, 6, 7,
                /* right  */  8, 9,10,   8,10,11,
                /* top    */ 12,13,14,  12,14,15,
                /* bottom */ 16,17,18,  16,18,19,
            };
            GL_CALL(glGenBuffers(1, &box_sides_ibo));
            GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_sides_ibo));
            GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                sizeof(idx), idx, GL_STATIC_DRAW));
        }

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

        {
            auto og = output->get_layout_geometry();
            float aspect = (float)og.width / og.height;
            animation.projection = glm::perspective(45.0f, aspect, 0.1f, 100.0f);
        }
    }


glm::vec2 world_to_screen_uv(const glm::vec3& world_pos,
                                  const glm::mat4& vp,
                                  float w, float h)
    {
        glm::vec4 clip = vp * glm::vec4(world_pos, 1.0f);
        if (std::abs(clip.w) < 1e-6f) return {0.5f, 0.5f};
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return { ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f };
    }


    void draw_light_orb(const glm::mat4& vp, const glm::vec3& orb_pos,
                        const glm::vec3& color, float intensity)
    {
        /* Billboard: always face camera */
        glm::vec3 cam = zoom_anim.current_pos;
        glm::vec3 forward = glm::normalize(cam - orb_pos);
        glm::vec3 up(0, 1, 0);
        glm::vec3 right = glm::normalize(glm::cross(up, forward));
        up = glm::cross(forward, right);

        glm::mat4 billboard(1.0f);
        billboard[0] = glm::vec4(right, 0);
        billboard[1] = glm::vec4(up, 0);
        billboard[2] = glm::vec4(forward, 0);
        billboard[3] = glm::vec4(orb_pos, 1);

        orb_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, background_vbo));
        orb_program.attrib_pointer("position", 2, 0, nullptr);
        orb_program.uniformMatrix4f("MVP", vp * billboard);
        orb_program.uniform1f("u_size", 0.4f);
        orb_program.uniform3f("u_color", color.r, color.g, color.b);
        orb_program.uniform1f("u_time", elapsed);
        orb_program.uniform1f("u_intensity", intensity);

        GL_CALL(glDepthMask(GL_FALSE));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE)); /* additive */
        GL_CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        GL_CALL(glDepthMask(GL_TRUE));

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        orb_program.deactivate();
    }

    void draw_neon_edges(const glm::mat4& mvp, const glm::vec3& color)
    {
        neon_edge_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, box_sides_vbo));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_sides_ibo));

        const GLsizei stride = 6 * sizeof(GLfloat);
        neon_edge_program.attrib_pointer("a_position", 3, stride, (void*)0);
        neon_edge_program.attrib_pointer("a_normal",   3, stride,
                                         (void*)(3 * sizeof(GLfloat)));

        neon_edge_program.uniformMatrix4f("MVP", mvp);
        neon_edge_program.uniform1f("u_expand", 0.012f);
        neon_edge_program.uniform3f("u_color", color.r, color.g, color.b);
        neon_edge_program.uniform1f("u_time", elapsed);
        neon_edge_program.uniform1f("u_intensity", 2.0f);

        /* Draw expanded outline with front-face culling (shows only edges) */
        GL_CALL(glEnable(GL_CULL_FACE));
        GL_CALL(glCullFace(GL_FRONT));
        GL_CALL(glDepthMask(GL_FALSE));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE)); /* additive */

        GL_CALL(glDrawElements(GL_TRIANGLES, 30, GL_UNSIGNED_INT, nullptr));

        GL_CALL(glDisable(GL_CULL_FACE));
        GL_CALL(glDepthMask(GL_TRUE));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        neon_edge_program.deactivate();
    }

    void fullscreen_quad(OpenGL::program_t& prog)
    {
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, background_vbo));
        prog.attrib_pointer("position", 2, 0, nullptr);
        GL_CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

   void draw_light_disc(const glm::mat4& vp, const glm::vec3& pos,
                         const glm::vec3& color, float size)
    {
        glm::vec3 cam = zoom_anim.current_pos;
        glm::vec3 forward = glm::normalize(cam - pos);
        glm::vec3 up(0, 1, 0);
        glm::vec3 right = glm::normalize(glm::cross(up, forward));
        up = glm::cross(forward, right);

        glm::mat4 billboard(1.0f);
        billboard[0] = glm::vec4(right, 0);
        billboard[1] = glm::vec4(up, 0);
        billboard[2] = glm::vec4(forward, 0);
        billboard[3] = glm::vec4(pos, 1);

        light_disc_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, background_vbo));
        light_disc_program.attrib_pointer("position", 2, 0, nullptr);
        light_disc_program.uniformMatrix4f("MVP", vp * billboard);
        light_disc_program.uniform1f("u_size", size);
        light_disc_program.uniform3f("u_color", color.r, color.g, color.b);
        light_disc_program.uniform1f("u_time", elapsed);

        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE));
        GL_CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        light_disc_program.deactivate();
    } 

void run_godrays_pass(const glm::vec2& light_uv, float intensity)
    {
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_godrays_fbo));
        GL_CALL(glViewport(0, 0, pp_width, pp_height));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        godrays_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_occlude_tex));  /* <-- occlusion */
        godrays_program.uniform2f("u_light_screen", light_uv.x, light_uv.y);
        godrays_program.uniform1f("u_density", 1.0f);
        godrays_program.uniform1f("u_weight", 0.25f);
        godrays_program.uniform1f("u_decay", 0.97f);
        godrays_program.uniform1f("u_exposure", 0.5f);
        godrays_program.uniform1f("u_intensity", intensity);
        fullscreen_quad(godrays_program);
        godrays_program.deactivate();
    }

    void run_bloom_pass()
    {
        int bw = pp_width / 2, bh = pp_height / 2;

        /* Extract bright pixels */
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_bloom_fbo[0]));
        GL_CALL(glViewport(0, 0, bw, bh));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        bloom_extract_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_scene_tex));
        bloom_extract_program.uniform1f("u_threshold", 0.45f);
        fullscreen_quad(bloom_extract_program);
        bloom_extract_program.deactivate();

        /* Horizontal blur */
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_bloom_fbo[1]));
        GL_CALL(glViewport(0, 0, bw, bh));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        blur_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[0]));
        blur_program.uniform2f("u_direction", 1.0f / bw, 0.0f);
        fullscreen_quad(blur_program);
        blur_program.deactivate();

        /* Vertical blur */
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_bloom_fbo[0]));
        GL_CALL(glViewport(0, 0, bw, bh));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        blur_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[1]));
        blur_program.uniform2f("u_direction", 0.0f, 1.0f / bh);
        fullscreen_quad(blur_program);
        blur_program.deactivate();

        /* Second blur pass for wider bloom */
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_bloom_fbo[1]));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        blur_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[0]));
        blur_program.uniform2f("u_direction", 2.0f / bw, 0.0f);
        fullscreen_quad(blur_program);
        blur_program.deactivate();

        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_bloom_fbo[0]));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        blur_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[1]));
        blur_program.uniform2f("u_direction", 0.0f, 2.0f / bh);
        fullscreen_quad(blur_program);
        blur_program.deactivate();
    }

    void run_composite_pass(GLint original_fbo, int w, int h,
                            float godray_strength, float bloom_strength)
    {
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, original_fbo));
        GL_CALL(glViewport(0, 0, w, h));

        composite_program.use(wf::TEXTURE_TYPE_RGBA);

        GL_CALL(glActiveTexture(GL_TEXTURE0));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_scene_tex));
        composite_program.uniform1i("u_scene", 0);

        GL_CALL(glActiveTexture(GL_TEXTURE1));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_godrays_tex));
        composite_program.uniform1i("u_godrays", 1);

        GL_CALL(glActiveTexture(GL_TEXTURE2));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[0]));
        composite_program.uniform1i("u_bloom", 2);

        composite_program.uniform1f("u_godray_strength", godray_strength);
        composite_program.uniform1f("u_bloom_strength", bloom_strength);
        composite_program.uniform1f("u_vignette", 0.8f);

        fullscreen_quad(composite_program);

        GL_CALL(glActiveTexture(GL_TEXTURE0));
        composite_program.deactivate();
    }
    /* ═════════════════════════════════════════════════════════════════════
     *  OPT C: BACKGROUND TEXTURE CACHE
     * ═════════════════════════════════════════════════════════════════════*/
    void ensure_bg_cache(int w, int h)
    {
        if (bg_cache_valid && bg_cache_w == w && bg_cache_h == h) return;

        if (!bg_cache_tex) {
            GL_CALL(glGenTextures(1, &bg_cache_tex));
            GL_CALL(glGenFramebuffers(1, &bg_cache_fbo));
        }
        GL_CALL(glBindTexture(GL_TEXTURE_2D, bg_cache_tex));
        GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

        GLint prev_fbo;
        GL_CALL(glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo));
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, bg_cache_fbo));
        GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, bg_cache_tex, 0));
        GL_CALL(glViewport(0, 0, w, h));

        background_program.use(wf::TEXTURE_TYPE_RGBA);
        background_program.uniform1f("u_time", 0.0f);
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, background_vbo));
        background_program.attrib_pointer("position", 2, 0, nullptr);
        GL_CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        background_program.deactivate();

        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo));

        bg_cache_w = w;
        bg_cache_h = h;
        bg_cache_valid = true;
    }

    void render_background_cached(int w, int h)
    {
        ensure_bg_cache(w, h);

        blit_program.use(wf::TEXTURE_TYPE_RGBA);
        GL_CALL(glBindTexture(GL_TEXTURE_2D, bg_cache_tex));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, background_vbo));
        blit_program.attrib_pointer("position", 2, 0, nullptr);
        GL_CALL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        blit_program.deactivate();
    }

    /* ═════════════════════════════════════════════════════════════════════
     *  RENDER: DRAW A SINGLE BOX'S COLORED SIDES + BACK
     * ═════════════════════════════════════════════════════════════════════*/
struct DynamicLight {
    glm::vec3 cam_pos{0, 0, 5};
    glm::vec3 light_pos{0, 0, 5};
    glm::vec3 light_color{1.0f, 0.95f, 0.9f};
    glm::vec3 accent_pos{0, 0, 2};
    glm::vec3 accent_color{0.4f, 0.6f, 1.0f};
    float time = 0.0f;
};

void draw_box_sides(const glm::mat4& mvp, const glm::mat4& model,
                    float brightness,
                    const glm::vec3& color1, const glm::vec3& color2,
                    const DynamicLight& light)
{

    GL_CALL(glEnable(GL_CULL_FACE));
        GL_CALL(glCullFace(GL_BACK));

    box_color_program.use(wf::TEXTURE_TYPE_RGBA);
    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, box_sides_vbo));
    GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_sides_ibo));

    const GLsizei stride = 6 * sizeof(GLfloat);
    box_color_program.attrib_pointer("a_position", 3, stride, (void*)0);
    box_color_program.attrib_pointer("a_normal",   3, stride,
                                     (void*)(3 * sizeof(GLfloat)));

    box_color_program.uniformMatrix4f("MVP", mvp);
    box_color_program.uniformMatrix4f("u_model", model);

    /* Normal matrix = transpose(inverse(upper-left 3×3 of model)) */
    glm::mat3 normal_mat = glm::mat3(glm::transpose(glm::inverse(model)));
    /* uniformMatrix3f helper — if your OpenGL::program_t doesn't have one,
       fall back to raw glUniformMatrix3fv */
    GLint loc = glGetUniformLocation(
        box_color_program.get_program_id(wf::TEXTURE_TYPE_RGBA),
        "u_normal_matrix");
    if (loc >= 0) {
        GL_CALL(glUniformMatrix3fv(loc, 1, GL_FALSE, &normal_mat[0][0]));
    }

    box_color_program.uniform3f("u_color1", color1.r, color1.g, color1.b);
    box_color_program.uniform3f("u_color2", color2.r, color2.g, color2.b);
    box_color_program.uniform1f("u_brightness", brightness);

    box_color_program.uniform3f("u_cam_pos",
        light.cam_pos.x, light.cam_pos.y, light.cam_pos.z);
    box_color_program.uniform3f("u_light_pos",
        light.light_pos.x, light.light_pos.y, light.light_pos.z);
    box_color_program.uniform3f("u_light_color",
        light.light_color.x, light.light_color.y, light.light_color.z);
    box_color_program.uniform3f("u_accent_pos",
        light.accent_pos.x, light.accent_pos.y, light.accent_pos.z);
    box_color_program.uniform3f("u_accent_color",
        light.accent_color.x, light.accent_color.y, light.accent_color.z);
    box_color_program.uniform1f("u_time", light.time);

        /* Draw all side + back faces */
        GL_CALL(glDrawElements(GL_TRIANGLES, sides_index_count,
            GL_UNSIGNED_INT, nullptr));

    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    box_color_program.deactivate();
     GL_CALL(glDisable(GL_CULL_FACE));
}

    /* ═════════════════════════════════════════════════════════════════════
     *  RENDER PASSES — 3D BOX VERSION
     * ═════════════════════════════════════════════════════════════════════*/
    void render_cursor_indicator(const glm::mat4& vp)
    {
        if (!cursor_indicator.active) return;
        if (cursor_program.get_program_id(wf::TEXTURE_TYPE_RGBA)==0||!cursor_vbo) return;
        int wx=cursor_indicator.workspace_x;
        int wy=cursor_indicator.workspace_y;
        float vo=-(float)wy*(1.0f+GRID_GAP);
        auto m=calculate_model_matrix(wx,vo);
        float lx=cursor_indicator.uv_position.x-0.5f;
        float ly=cursor_indicator.uv_position.y-0.5f;
        m=glm::translate(m,glm::vec3(lx,ly,0.02f));
        m=glm::scale(m,glm::vec3(0.04f*cached_aspect,0.04f,1.0f));
        cursor_program.use(wf::TEXTURE_TYPE_RGBA);
        cursor_program.uniformMatrix4f("mvp",vp*m);
        cursor_program.uniform1f("u_time",elapsed);
        cursor_program.uniform3f("u_color",1.0f,1.0f,1.0f);
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER,cursor_vbo));
        cursor_program.attrib_pointer("position",2,0,nullptr);
        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN,0,50));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        cursor_program.deactivate();
    }

    /* ── Main draw loop — 3D BOX MESHES ────────────────────────────────── */
    using WsRenderState = cube_render_node_t::WsRenderState;

void draw_box_silhouette(const glm::mat4& mvp)
    {
        silhouette_program.use(wf::TEXTURE_TYPE_RGBA);

        /* Front face silhouette */
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, panel_vbo));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_ibo));
        silhouette_program.attrib_pointer("a_position", 2,
            4*sizeof(GLfloat), (void*)0);
        silhouette_program.uniformMatrix4f("MVP", mvp);
        GL_CALL(glDrawElements(GL_TRIANGLES, front_index_count,
            GL_UNSIGNED_INT, nullptr));

        /* Sides + back silhouette */
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, box_sides_vbo));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_sides_ibo));
        silhouette_program.attrib_pointer("a_position", 3,
            6*sizeof(GLfloat), (void*)0);
        silhouette_program.uniformMatrix4f("MVP", mvp);
        GL_CALL(glDrawElements(GL_TRIANGLES, sides_index_count,
            GL_UNSIGNED_INT, nullptr));

        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        silhouette_program.deactivate();
    }

   void render_from_states(const wf::scene::render_instruction_t& data,
        std::vector<WsRenderState>& states, int gw, int gh)
    {
        if (skip_first_render > 0) {
            skip_first_render--;
            return;
        }

        data.pass->custom_gles_subpass([&]
            {
            if (program.get_program_id(wf::TEXTURE_TYPE_RGBA)==0)
                load_program();

            /* ─── Save original FBO + viewport ─────────────────────── */
            GLint original_fbo;
            GLint original_viewport[4];
            GL_CALL(glGetIntegerv(GL_FRAMEBUFFER_BINDING, &original_fbo));
            GL_CALL(glGetIntegerv(GL_VIEWPORT, original_viewport));

            int vp_w = original_viewport[2];
            int vp_h = original_viewport[3];

            auto og = output->get_layout_geometry();
            float ow = (float)og.width, oh = (float)og.height;

            ensure_postprocess_fbos(vp_w, vp_h);

            update_cached_grid_metrics();
            rebuild_model_cache(gw, gh);

            auto vp = calculate_vp_matrix(data.target);

            /* ─── Scatter animation state ──────────────────────────── */
            float oz    = cached_overview_z;
            float cz    = zoom_anim.current_pos.z;
            float close = calculate_close_z();
            float t = std::clamp((oz - cz) / (oz - close), 0.0f, 1.0f);
            float max_dist = std::sqrt(
                (float)(gw * gw) + (float)(gh * gh));

            /* ─── Build dynamic lighting ───────────────────────────── */
            DynamicLight dyn_light;
            dyn_light.time = elapsed;
            dyn_light.cam_pos = zoom_anim.current_pos;
            dyn_light.light_pos = dyn_light.cam_pos + glm::vec3(0.5f, 1.0f, 0.3f);
            dyn_light.light_color = glm::vec3(1.0f, 0.95f, 0.9f);

            auto sel_center = workspace_grid_center(
                selected_ws.x, selected_ws.y);

            glm::vec3 sel_c1, sel_c2;
            workspace_colors(selected_ws.x, selected_ws.y, gw,
                             sel_c1, sel_c2);
            dyn_light.accent_color = glm::mix(sel_c1, sel_c2, 0.5f) * 1.5f;

            /* Project mouse cursor to world XY at the orb Z plane */
            float orb_z = -0.2f;  /* stays behind the cubes */
            Ray mouse_ray = screen_to_world_ray(
                last_cursor_pos.x, last_cursor_pos.y, output);

            glm::vec3 orb_pos;
            if (std::abs(mouse_ray.dir.z) > 1e-6f) {
                float t_hit = (orb_z - mouse_ray.origin.z) / mouse_ray.dir.z;
                orb_pos = mouse_ray.origin + t_hit * mouse_ray.dir;
                orb_pos.z = orb_z;
            } else {
                orb_pos = glm::vec3(sel_center.x, sel_center.y, orb_z);
            }
            dyn_light.accent_pos = orb_pos;

            /* ─── Pre-compute all scatter matrices ─────────────────── */
            int target_idx = -1;
            struct PanelDraw {
                int idx, wx, wy;
                float depth, bright;
                glm::mat4 scatter_model, scatter_mvp;
            };
            std::vector<PanelDraw> panels;
            panels.reserve(gw * gh);

            for (int wy = 0; wy < gh; wy++) {
                for (int wx = 0; wx < gw; wx++) {
                    int idx = wy * gw + wx;
                    bool is_target = (wx == zoom_target_ws.x &&
                                      wy == zoom_target_ws.y);
                    if (is_target) { target_idx = idx; continue; }

                    float dx = (float)wx - (float)zoom_target_ws.x;
                    float dy = (float)wy - (float)zoom_target_ws.y;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    float norm_dist = dist / std::max(max_dist, 1.0f);
                    float delay = norm_dist * 0.3f;
                    float local_t = std::clamp(
                        (t - delay) / (1.0f - delay), 0.0f, 1.0f);
                    float eased_t = local_t * local_t;
                    float pdepth = eased_t * (1.0f + dist * 0.6f);

                    const glm::mat4& model = cached_models[idx];
                    glm::mat4 sm = model;
                    float bright = (wx == selected_ws.x &&
                                    wy == selected_ws.y)
                                   ? 1.0f : 0.75f;

                    if (t > 0.01f) {
                        float ddx = dx, ddy = dy;
                        if (dist < 0.01f) { ddx = 0; ddy = 0; }
                        else { ddx /= dist; ddy /= dist; }
                        float slide = eased_t * (1.5f + dist * 0.8f);
                        sm = glm::translate(model,
                            glm::vec3(ddx * slide, -ddy * slide, -pdepth));
                        float fade = eased_t * (0.5f + norm_dist * 0.4f);
                        bright *= std::max(0.0f, 1.0f - fade);
                    }

                               float swing_phase = (float)(wx * 3 + wy * 7) * 1.618f;
                    float swing_angle = glm::radians(17.5f) *
                        std::sin(elapsed * 0.4f + swing_phase) * (1.0f - t);
                    sm = glm::rotate(sm, swing_angle, glm::vec3(0.0f, 1.0f, 0.0f));

                    

                    panels.push_back({idx, wx, wy, pdepth, bright,
                                      sm, vp * sm});
                }
            }

            std::sort(panels.begin(), panels.end(),
                [](const PanelDraw& a, const PanelDraw& b) {
                    return a.depth > b.depth;
                });

            /* ═══════════════════════════════════════════════════════
             *  OCCLUSION PASS: bright light + black cube silhouettes
             *  This is what the god rays sample from
             * ═══════════════════════════════════════════════════════*/
            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_occlude_fbo));
            GL_CALL(glViewport(0, 0, vp_w, vp_h));
            GL_CALL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
            GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

            GL_CALL(glEnable(GL_BLEND));
            GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
            GL_CALL(glDisable(GL_DEPTH_TEST));

            /* Draw a large bright disc at the light position */
            draw_light_disc(vp, orb_pos, dyn_light.accent_color, 1.8f);

            /* Also a secondary one at the camera light */
            draw_light_disc(vp, dyn_light.light_pos,
                            glm::vec3(0.8f, 0.75f, 0.6f), 1.0f);

            /* Now stamp black silhouettes of all cubes on top */
            GL_CALL(glEnable(GL_DEPTH_TEST));
            GL_CALL(glDepthFunc(GL_LESS));
            GL_CALL(glDepthMask(GL_TRUE));
            GL_CALL(glDisable(GL_BLEND));

            for (auto& pd : panels) {
                if (projected_quad_size_px(pd.scatter_mvp, ow, oh) < CULL_PX)
                    continue;
                draw_box_silhouette(pd.scatter_mvp);
            }
            if (target_idx >= 0) {
                glm::mat4 tmodel = cached_models[target_idx];
                float swing_phase = (float)(zoom_target_ws.x * 3 +
                                            zoom_target_ws.y * 7) * 1.618f;
                float swing_angle = glm::radians(17.5f) *
                        std::sin(elapsed * 0.4f + swing_phase) * (1.0f - t);
                tmodel = glm::rotate(tmodel, swing_angle,
                                     glm::vec3(0.0f, 1.0f, 0.0f));
                glm::mat4 tmvp = vp * tmodel;
                draw_box_silhouette(tmvp);
            }

            GL_CALL(glDisable(GL_DEPTH_TEST));

            /* ═══════════════════════════════════════════════════════
             *  GOD RAYS PASS: radial blur the occlusion texture
             * ═══════════════════════════════════════════════════════*/
            glm::vec2 orb_screen_uv = world_to_screen_uv(orb_pos, vp, ow, oh);
            float edge_dist = std::min({orb_screen_uv.x, 1.0f - orb_screen_uv.x,
                                        orb_screen_uv.y, 1.0f - orb_screen_uv.y});
            float ray_intensity = std::clamp(edge_dist * 5.0f, 0.0f, 1.0f);

            run_godrays_pass(orb_screen_uv, ray_intensity);

            /* ═══════════════════════════════════════════════════════
             *  SCENE PASS: background + god rays + cubes + orbs
             * ═══════════════════════════════════════════════════════*/
            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, pp_scene_fbo));
            GL_CALL(glViewport(0, 0, vp_w, vp_h));
            GL_CALL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
            GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

            GL_CALL(glEnable(GL_BLEND));
            GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

            /* Draw background */
            render_background_cached(vp_w, vp_h);

            /* Blit god rays additively on top of background */
            GL_CALL(glBlendFunc(GL_ONE, GL_ONE));
            GL_CALL(glDepthMask(GL_FALSE));
            blit_program.use(wf::TEXTURE_TYPE_RGBA);
            GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_godrays_tex));
            fullscreen_quad(blit_program);
            blit_program.deactivate();
            GL_CALL(glDepthMask(GL_TRUE));
            GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

            /* Draw the visible light orbs */
      //      draw_light_orb(vp, orb_pos, dyn_light.accent_color, 1.5f);
        //    draw_light_orb(vp, dyn_light.light_pos,
          //                 glm::vec3(1.0f, 0.9f, 0.8f), 0.6f);

            /* ─── Draw cubes on top ────────────────────────────────── */
            GL_CALL(glEnable(GL_DEPTH_TEST));
            GL_CALL(glDepthFunc(GL_LESS));
            GL_CALL(glDepthMask(GL_TRUE));

            for (auto& pd : panels) {
                auto& st = states[pd.idx];
                if (!st.allocated) continue;
                if (projected_quad_size_px(pd.scatter_mvp, ow, oh) < CULL_PX)
                    continue;

                glm::vec3 c1, c2;
                workspace_colors(pd.wx, pd.wy, gw, c1, c2);

                draw_box_sides(pd.scatter_mvp, pd.scatter_model,
                               pd.bright, c1, c2, dyn_light);

                if (pd.wx == selected_ws.x && pd.wy == selected_ws.y) {
             //       draw_neon_edges(pd.scatter_mvp, dyn_light.accent_color);
                }

               float face_bright = ((pd.wx == selected_ws.x &&
                                      pd.wy == selected_ws.y) ||
                                     (pd.wx == zoom_target_ws.x &&
                                      pd.wy == zoom_target_ws.y))
                                    ? 1.3f : 1.30f;

                GL_CALL(glBlendFunc(GL_ONE, GL_ZERO));
                program.use(wf::TEXTURE_TYPE_RGBA);
                GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, panel_vbo));
                GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_ibo));
                program.attrib_pointer("position",   2,
                    4*sizeof(GLfloat), (void*)0);
                program.attrib_pointer("uvPosition", 2,
                    4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)));
                auto tex = wf::gles_texture_t::from_aux(st.fb);
                GL_CALL(glBindTexture(GL_TEXTURE_2D, tex.tex_id));
                program.uniformMatrix4f("MVP", pd.scatter_mvp);
                program.uniform1f("u_brightness", face_bright);
                GL_CALL(glDrawElements(GL_TRIANGLES, front_index_count, GL_UNSIGNED_INT, nullptr));
                GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
                GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
                GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
                program.deactivate();
            }

            /* ─── Target panel ─────────────────────────────────────── */
            if (target_idx >= 0) {
                auto& st = states[target_idx];
                if (st.allocated) {
                   
                    glm::mat4 model = cached_models[target_idx];

                    float swing_phase = (float)(zoom_target_ws.x * 3 +
                                                zoom_target_ws.y * 7) * 1.618f;
                    float swing_angle = glm::radians(17.5f) *
                        std::sin(elapsed * 0.4f + swing_phase) * (1.0f - t);
                    model = glm::rotate(model, swing_angle,
                                        glm::vec3(0.0f, 1.0f, 0.0f));

                    glm::mat4 mvp = vp * model;

                    glm::vec3 c1, c2;
                    workspace_colors(zoom_target_ws.x, zoom_target_ws.y,
                                     gw, c1, c2);

                    draw_box_sides(mvp, model, 1.0f, c1, c2, dyn_light);
               //     draw_neon_edges(mvp, dyn_light.accent_color);

                   /* Front face — raw desktop, no lighting */
                GL_CALL(glBlendFunc(GL_ONE, GL_ZERO));  /* fully opaque, no blending */

                program.use(wf::TEXTURE_TYPE_RGBA);
                GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, panel_vbo));
                GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_ibo));
                program.attrib_pointer("position",   2,
                    4*sizeof(GLfloat), (void*)0);
                program.attrib_pointer("uvPosition", 2,
                    4*sizeof(GLfloat), (void*)(2*sizeof(GLfloat)));
                auto tex = wf::gles_texture_t::from_aux(st.fb);
                GL_CALL(glBindTexture(GL_TEXTURE_2D, tex.tex_id));
                program.uniformMatrix4f("MVP",mvp);  /* or mvp for target */
                GL_CALL(glEnable(GL_POLYGON_OFFSET_FILL));
                GL_CALL(glPolygonOffset(-1.0f, -1.0f));
                GL_CALL(glDrawElements(GL_TRIANGLES, front_index_count, GL_UNSIGNED_INT, nullptr));

                GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));  /* restore */

                GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
                GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
                program.deactivate();
                }
            }

            GL_CALL(glDisable(GL_DEPTH_TEST));
            GL_CALL(glDisable(GL_BLEND));

            /* ═══════════════════════════════════════════════════════
             *  BLOOM PASS
             * ═══════════════════════════════════════════════════════*/
            run_bloom_pass();

            /* ═══════════════════════════════════════════════════════
             *  FINAL COMPOSITE
             * ═══════════════════════════════════════════════════════*/
            GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, original_fbo));
            GL_CALL(glViewport(original_viewport[0], original_viewport[1],
                               original_viewport[2], original_viewport[3]));

            GL_CALL(glDisable(GL_DEPTH_TEST));
            GL_CALL(glEnable(GL_BLEND));

composite_program.use(wf::TEXTURE_TYPE_RGBA);

            GL_CALL(glActiveTexture(GL_TEXTURE0));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_scene_tex));
            composite_program.uniform1i("u_scene", 0);

            GL_CALL(glActiveTexture(GL_TEXTURE1));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_godrays_tex));
            composite_program.uniform1i("u_godrays", 1);

            GL_CALL(glActiveTexture(GL_TEXTURE2));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_bloom_tex[0]));
            composite_program.uniform1i("u_bloom", 2);

            GL_CALL(glActiveTexture(GL_TEXTURE3));
            GL_CALL(glBindTexture(GL_TEXTURE_2D, pp_occlude_tex));
            composite_program.uniform1i("u_occlude", 3);

            composite_program.uniform1f("u_godray_strength", 0.5f);
            composite_program.uniform1f("u_bloom_strength", 0.5f);
            composite_program.uniform1f("u_vignette", 0.8f);

            fullscreen_quad(composite_program);

            GL_CALL(glActiveTexture(GL_TEXTURE0));
            composite_program.deactivate();

            GL_CALL(glDisable(GL_BLEND));
        });
    }
    /* ═════════════════════════════════════════════════════════════════════
     *  LIFECYCLE / SIGNALS / INPUT
     * ═════════════════════════════════════════════════════════════════════*/

    wf::signal::connection_t<cube_control_signal> on_cube_control =
        [=](cube_control_signal *d)
    {
        rotate_and_zoom_cube(d->angle,d->zoom,d->ease,d->last_frame);
        d->carried_out = true;
    };

    void rotate_and_zoom_cube(double a, double z, double e, bool last)
    {
        if (last) { deactivate(); return; }
        if (!activate()) return;
        float oz=calculate_overview_z();
        animation.cube_animation.rotation.set(a,a);
        animation.cube_animation.zoom.set(z,z);
        animation.cube_animation.offset_z.set(oz,oz);
        animation.cube_animation.start();
        update_view_matrix();
        output->render->schedule_redraw();
    }

public:
    void init() override
    {
        input_grab = std::make_unique<wf::input_grab_t>(
            "cube", output, nullptr, this, nullptr);
        input_grab->set_wants_raw_input(true);
        animation.cube_animation.offset_y.set(0,0);
        animation.cube_animation.offset_z.set(0,0);
        animation.cube_animation.rotation.set(0,0);
        animation.cube_animation.zoom.set(1,1);
        animation.cube_animation.ease_deformation.set(0,0);
        animation.cube_animation.max_tilt.set(0,0);
        animation.cube_animation.start();
        camera_y_offset.set(0,0);
        popout_scale_animation.set(1,1);
        zoom_anim.snap(glm::vec3(0,0,0));
        reload_background();
        output->connect(&on_cube_control);
        wf::gles::run_in_context([&]{ load_program(); });
    }

    bool activate()
    {
        if (output->is_plugin_active(grab_interface.name)) return true;
        if (!output->activate_plugin(&grab_interface)) return false;

        wf::get_core().connect(&on_motion_event);
        workspace_already_set = false;
        last_frame_time = std::chrono::steady_clock::now();
        origin_ws   = output->wset()->get_current_workspace();
        selected_ws = origin_ws;
        prev_selected_ws = {-1, -1};
        zoom_state  = ZoomState::OVERVIEW;

        output->wset()->set_workspace({0,0});
        render_node = std::make_shared<cube_render_node_t>(this);
        wf::scene::add_front(wf::get_core().scene(), render_node);
        output->render->add_effect(&pre_hook, wf::OUTPUT_EFFECT_PRE);
        output->wset()->set_workspace({0,0});
        input_grab->grab_input(wf::scene::layer::OVERLAY);
        update_cached_grid_metrics();
        models_dirty = true;
        raycast_cache_dirty = true;

        auto wsize=output->wset()->get_workspace_grid_size();
        animation.side_angle=2*(float)M_PI/(float)wsize.width;
        identity_z_offset=0.5f/std::tan(animation.side_angle/2);
        if (wsize.width==1) identity_z_offset=0.0f;
        plane_z=identity_z_offset;

        float oz=calculate_overview_z(), cz=calculate_close_z();
        auto oc=workspace_grid_center(origin_ws.x, origin_ws.y);
        zoom_anim.snap(glm::vec3(oc.x,oc.y,cz));
        startup_frames = 3;
        skip_first_render = 3;

        animation.cube_animation.zoom.set(1,1);
        animation.cube_animation.rotation.set(0,0);
        animation.cube_animation.offset_y.set(0,0);
        animation.cube_animation.offset_z.set(oz,oz);
        animation.cube_animation.ease_deformation.set(0,0);
        animation.cube_animation.max_tilt.set(0,0);
        animation.cube_animation.start();
        {
            auto og = output->get_layout_geometry();
            float aspect = (float)og.width / og.height;
            animation.projection = glm::perspective(45.0f, aspect, 0.1f, 100.0f);
        }
        reload_background();
        output->render->damage_whole();
        return true;
    }

    int calculate_viewport_dx_from_rotation() { return 0; }
    int calculate_viewport_dy_from_camera()   { return 0; }

    void deactivate()
    {
        if (!output->is_plugin_active(grab_interface.name)) return;
        is_dragging_window = false;
        dragged_view       = nullptr;

        wf::scene::remove_child(render_node);
        output->render->damage_whole();
        render_node = nullptr;
        output->render->rem_effect(&pre_hook);
        input_grab->ungrab_input();
        output->deactivate_plugin(&grab_interface);
        wf::get_core().unhide_cursor();
        on_motion_event.disconnect();

        if (workspace_already_set && right_click_target_ws.x>=0)
            output->wset()->set_workspace(right_click_target_ws);
        else
            output->wset()->set_workspace(selected_ws);

        workspace_already_set = false;
        right_click_target_ws = {-1,-1};
        sync_cam_y_to_z       = false;
        has_virtual_hit       = false;
        zoom_state            = ZoomState::OVERVIEW;
        zoom_anim.snap(glm::vec3(0,0,0));
    }

    bool move_vp(int dir)
    {
        if (!activate()) return false;
        animation.in_exit = false;
        if (is_zoomed_in()||is_zooming()) zoom_out_to_overview();
        auto grid=output->wset()->get_workspace_grid_size();
        selected_ws.x=(selected_ws.x+dir+grid.width)%grid.width;
        output->render->schedule_redraw();
        return true;
    }

    bool move_vp_vertical(int dir)
    {
        bool was=output->is_plugin_active(grab_interface.name);
        if (!was && !activate()) return false;
        animation.in_exit = false;
        if (is_zoomed_in()||is_zooming()) zoom_out_to_overview();
        auto grid=output->wset()->get_workspace_grid_size();
        selected_ws.y=(selected_ws.y+dir+grid.height)%grid.height;
        output->render->schedule_redraw();
        return true;
    }

    bool input_grabbed()
    {
        bool was=output->is_plugin_active(grab_interface.name);
        if (!activate()) return false;
        animation.in_exit=false;
        wf::get_core().unhide_cursor();
        if (was) {
            if      (is_zooming())   {}
            else if (is_zoomed_in()) zoom_out_to_overview();
            else                     zoom_into_workspace(selected_ws);
        }
        update_view_matrix();
        output->render->schedule_redraw();
        return false;
    }

    void input_ungrabbed()
    {
        workspace_already_set=true;
        right_click_target_ws=selected_ws;
        zoom_target_ws=selected_ws;
        auto center=workspace_grid_center(selected_ws.x,selected_ws.y);
        zoom_anim.start(zoom_anim.current_pos,
            glm::vec3(center.x,center.y,calculate_close_z()),0.35f);
        animation.in_exit=true;
    }

    void handle_pointer_button(const wlr_pointer_button_event& ev) override
    {
        if (ev.button==BTN_LEFT) {
            if (ev.state==WL_POINTER_BUTTON_STATE_PRESSED) {
                auto cur=wf::get_core().get_cursor_position();
                last_cursor_pos=cur; drag_start_cursor=cur;
                wf::get_core().unhide_cursor();
                Ray ray=screen_to_world_ray(cur.x,cur.y,output);
                HitInfo hit=raycast_at_window_depth(ray.origin,ray.dir,output);
                if (hit.ws.x>=0) {
                    cursor_indicator={true,{},hit.local_uv,
                        hit.ws.x,hit.ws.y};
                    selected_ws=hit.ws;
                    auto bb=output->get_layout_geometry();
                    float vx=(hit.ws.x+hit.local_uv.x)*bb.width;
                    float vy=(hit.ws.y+hit.local_uv.y)*bb.height;
                    dragged_view=find_window_at_cursor_on_face({vx,vy},hit.ws);
                    dragged_window_face_index=hit.ws.x;
                    is_dragging_window=true;
                    if (dragged_view) {
                        drag_start_workspace=
                            output->wset()->get_view_main_workspace(dragged_view);
                        auto g=dragged_view->get_geometry();
                        drag_offset={vx-g.x,vy-g.y};
                        start_wobbly(dragged_view,vx,vy);
                        if (render_node) {
                            render_node->drag_active=true;
                            render_node->drag_window_geom=g;
                            render_node->drag_window_geom_prev=g;
                        }
                    }
                } else {
                    is_dragging_window=false;
                    cursor_indicator.active=false;
                }
            } else {
                if (is_dragging_window && dragged_view) {
                    end_wobbly(dragged_view);
                    auto ws_set=output->wset();
                    auto wg=dragged_view->get_geometry();
                    auto og=output->get_layout_geometry();
                    auto grid=ws_set->get_workspace_grid_size();
                    int cx=wg.x+wg.width/2, cy=wg.y+wg.height/2;
                    wf::point_t tw{
                        std::max(0,std::min(cx/og.width, grid.width-1)),
                        std::max(0,std::min(cy/og.height,grid.height-1))};
                    auto cw=ws_set->get_view_main_workspace(dragged_view);
                    if (tw!=cw) {
                        ws_set->move_to_workspace(dragged_view,tw);
                        wf::geometry_t ag=wg;
                        ag.x-=tw.x*og.width;
                        ag.y-=tw.y*og.height;
                        dragged_view->set_geometry(ag);
                    }
                    if (render_node) {
                        render_node->drag_active=false;
                        render_node->damage_all_workspace_windows();
                        render_node->damage_all_workspace_desktops();
                    }
                } else if (!dragged_view) {
                    auto cur=wf::get_core().get_cursor_position();
                    float dx=cur.x-drag_start_cursor.x;
                    float dy=cur.y-drag_start_cursor.y;
                    if (dx*dx+dy*dy<25.0f && !is_zooming()) {
                        if (is_zoomed_in()) {
                            zoom_out_to_overview();
                        } else {
                            workspace_already_set=true;
                            right_click_target_ws=selected_ws;
                            zoom_target_ws=selected_ws;
                            auto center=workspace_grid_center(
                                selected_ws.x,selected_ws.y);
                            zoom_anim.start(zoom_anim.current_pos,
                                glm::vec3(center.x,center.y,
                                    calculate_close_z()),5.0f);
                            animation.in_exit=true;
                        }
                    }
                }
                is_dragging_window=false;
                dragged_view=nullptr;
                dragged_window_face_index=-1;
                if (render_node) render_node->drag_active=false;
            }
        } else if (ev.button==BTN_RIGHT &&
                   ev.state==WL_POINTER_BUTTON_STATE_PRESSED) {
            if (is_zoomed_in()||is_zooming()) {
                workspace_already_set=true;
                right_click_target_ws=zoom_target_ws;
                selected_ws=zoom_target_ws;
                auto center=workspace_grid_center(
                    zoom_target_ws.x,zoom_target_ws.y);
                zoom_anim.start(zoom_anim.current_pos,
                    glm::vec3(center.x,center.y,calculate_close_z()),0.3f);
                animation.in_exit=true;
            } else {
                auto cur=wf::get_core().get_cursor_position();
                Ray ray=screen_to_world_ray(cur.x,cur.y,output);
                HitInfo hit=raycast_to_workspace(ray.origin,ray.dir,output);
                wf::point_t target=hit.ws.x>=0 ? hit.ws : selected_ws;
                workspace_already_set=true;
                right_click_target_ws=target;
                selected_ws=target;
                zoom_target_ws=target;
                auto center=workspace_grid_center(target.x,target.y);
                zoom_anim.start(zoom_anim.current_pos,
                    glm::vec3(center.x,center.y,calculate_close_z()),0.4f);
                animation.in_exit=true;
            }
            cursor_indicator.active=false;
        }
        output->render->schedule_redraw();
    }

    void handle_pointer_axis(const wlr_pointer_axis_event& ev) override
    {
        if (ev.orientation==WL_POINTER_AXIS_VERTICAL_SCROLL)
            pointer_scrolled(ev.delta);
    }

    void pointer_scrolled(double amount)
    {
        if (animation.in_exit) return;
        float oz=calculate_overview_z();
        float close_z=calculate_close_z();
        float cur_z=zoom_anim.current_pos.z;
        if (cur_z<0.1f) cur_z=(float)animation.cube_animation.offset_z;
        float speed=cur_z*0.1f*ZVelocity;
        float new_z=std::clamp(cur_z+(float)amount*speed,
                               close_z*0.9f,oz*2.5f);
        float t=std::clamp((oz-new_z)/(oz-close_z),0.0f,1.0f);
        auto center=workspace_grid_center(selected_ws.x,selected_ws.y);
        zoom_anim.snap(glm::vec3(center.x*t,center.y*t,new_z));
        animation.cube_animation.offset_z.set(new_z,new_z);
        if (t>1.0f) {
            zoom_target_ws=selected_ws;
            if (zoom_state!=ZoomState::ZOOMED_IN)
                zoom_state=ZoomState::ZOOMING_IN;
        } else if (t<0.1f) {
            if (zoom_state!=ZoomState::OVERVIEW)
                zoom_state=ZoomState::ZOOMING_OUT;
        } else {
            zoom_state=amount<0 ? ZoomState::ZOOMING_IN
                                : ZoomState::ZOOMING_OUT;
            zoom_target_ws=selected_ws;
        }
        update_zoom_state();
        output->render->schedule_redraw();
    }

    wf::signal::connection_t<wf::input_event_signal<wlr_pointer_motion_event>>
        on_motion_event =
            [=](wf::input_event_signal<wlr_pointer_motion_event> *ev)
    { pointer_moved(ev->event); };

    void pointer_moved(wlr_pointer_motion_event *ev)
    {
        if (animation.in_exit) return;
        auto cur=wf::get_core().get_cursor_position();

        float pdx = cur.x - last_cursor_pos.x;
        float pdy = cur.y - last_cursor_pos.y;
        if (!is_dragging_window &&
            (pdx*pdx + pdy*pdy) < POINTER_DEDUP_DIST_SQ)
            return;

        if (output->is_plugin_active(grab_interface.name)) {
            Ray ray=screen_to_world_ray(cur.x,cur.y,output);
            HitInfo hit=raycast_to_workspace(ray.origin,ray.dir,output);
            if (hit.ws.x>=0) {
                cursor_indicator={true,ray.origin+hit.t*ray.dir,
                    hit.local_uv,hit.ws.x,hit.ws.y};
                if (zoom_state==ZoomState::OVERVIEW && 
                    !is_dragging_window &&
                    !animation.in_exit)          // <── add this
                    selected_ws=hit.ws;
            }else {
                cursor_indicator.active=false;
            }
        }
        if (is_dragging_window && dragged_view) {
            Ray ray=screen_to_world_ray(cur.x,cur.y,output);
            HitInfo hit=raycast_at_window_depth(ray.origin,ray.dir,output);
            if (hit.ws.x>=0) {
                auto bb=output->get_layout_geometry();
                float vx=(hit.ws.x+hit.local_uv.x)*bb.width;
                float vy=(hit.ws.y+hit.local_uv.y)*bb.height;
                move_wobbly(dragged_view,vx,vy);
                wf::geometry_t ng=dragged_view->get_geometry();
                ng.x=(int)(vx-drag_offset.x);
                ng.y=(int)(vy-drag_offset.y);
                dragged_view->set_geometry(ng);
                if (render_node) {
                    render_node->drag_window_geom_prev=
                        render_node->drag_window_geom;
                    render_node->drag_active=true;
                    render_node->drag_window_geom=
                        dragged_view->get_geometry();
                }
            }
        }
        last_cursor_pos=cur;
        output->render->schedule_redraw();
    }

    /* ═════════════════════════════════════════════════════════════════════
     *  FIX 1: CONDITIONAL DAMAGE PRE_HOOK
     * ═════════════════════════════════════════════════════════════════════*/
    wf::effect_hook_t pre_hook = [=]()
    {
        auto now=std::chrono::steady_clock::now();
        float dt=std::chrono::duration<float>(now-last_frame_time).count();
        last_frame_time=now;
        elapsed+=dt;

        if (startup_frames > 0) {
            startup_frames--;
            if (startup_frames == 0) {
                float oz = calculate_overview_z();
                zoom_anim.start(zoom_anim.current_pos,
                                glm::vec3(0.0f, 0.0f, oz), 5.0f);
            }
            output->render->schedule_redraw();
        }

        bool zoom_was_running = zoom_anim.running;
        zoom_anim.update(dt);
        update_zoom_state();
        update_view_matrix();

        bool anim = animation.cube_animation.running() || zoom_anim.running
                    || zoom_was_running;

        wf::scene::damage_node(render_node, render_node->get_bounding_box());

        if      (anim)              output->render->schedule_redraw();
        else if (animation.in_exit) deactivate();
        else                        output->render->schedule_redraw();
    };

    void fini() override
    {
        if (output->is_plugin_active(grab_interface.name)) deactivate();
        wf::gles::run_in_context_if_gles([&]{

            program.free_resources();
            cap_program.free_resources();
            background_program.free_resources();
            cursor_program.free_resources();
            beam_program.free_resources();
            blit_program.free_resources();
            box_color_program.free_resources();
            if (cursor_vbo)     { GL_CALL(glDeleteBuffers(1,&cursor_vbo)); }
            if (background_vbo) { GL_CALL(glDeleteBuffers(1,&background_vbo)); }
            if (panel_vbo)      { GL_CALL(glDeleteBuffers(1,&panel_vbo)); }
            if (panel_ibo)      { GL_CALL(glDeleteBuffers(1,&panel_ibo)); }
            if (box_sides_vbo)  { GL_CALL(glDeleteBuffers(1,&box_sides_vbo)); }
            if (box_sides_ibo)  { GL_CALL(glDeleteBuffers(1,&box_sides_ibo)); }
            if (bg_cache_tex)   { GL_CALL(glDeleteTextures(1,&bg_cache_tex)); }
            if (bg_cache_fbo)   { GL_CALL(glDeleteFramebuffers(1,&bg_cache_fbo)); }
            if (pp_scene_depth_rbo) { GL_CALL(glDeleteRenderbuffers(1, &pp_scene_depth_rbo)); }
             godrays_program.free_resources();
            bloom_extract_program.free_resources();
            blur_program.free_resources();
            composite_program.free_resources();
            orb_program.free_resources();
            neon_edge_program.free_resources();
            if (pp_scene_tex)    { GL_CALL(glDeleteTextures(1, &pp_scene_tex)); }
            if (pp_scene_fbo)    { GL_CALL(glDeleteFramebuffers(1, &pp_scene_fbo)); }
            if (pp_godrays_tex)  { GL_CALL(glDeleteTextures(1, &pp_godrays_tex)); }
            if (pp_godrays_fbo)  { GL_CALL(glDeleteFramebuffers(1, &pp_godrays_fbo)); }
            for (int i = 0; i < 2; i++) {
                if (pp_bloom_tex[i]) GL_CALL(glDeleteTextures(1, &pp_bloom_tex[i]));
                if (pp_bloom_fbo[i]) GL_CALL(glDeleteFramebuffers(1, &pp_bloom_fbo[i]));
            }
        });
    }
};

/* ═══════════════════════════════════════════════════════════════════════════
 *  TOP-LEVEL WRAPPER
 * ═══════════════════════════════════════════════════════════════════════════*/
class WayfireVerticalExpo : public wf::plugin_interface_t,
    public wf::per_output_tracker_mixin_t<wayfire_cube>
{
    wf::ipc_activator_t rotate_left {"vertical_expo/rotate_left"};
    wf::ipc_activator_t rotate_right{"vertical_expo/rotate_right"};
    wf::ipc_activator_t rotate_up   {"vertical_expo/rotate_up"};
    wf::ipc_activator_t rotate_down {"vertical_expo/rotate_down"};
    wf::ipc_activator_t activate_   {"vertical_expo/activate"};

public:
    void init() override
    {
        if (!wf::get_core().is_gles2()) {
            LOGE("cube: requires GLES2"); return;
        }
        this->init_output_tracking();
        rotate_left .set_handler([=](wf::output_t *o, wayfire_view)
            { return output_instance[o]->move_vp(-1); });
        rotate_right.set_handler([=](wf::output_t *o, wayfire_view)
            { return output_instance[o]->move_vp(+1); });
        rotate_up   .set_handler([=](wf::output_t *o, wayfire_view)
            { return output_instance[o]->move_vp_vertical(-1); });
        rotate_down .set_handler([=](wf::output_t *o, wayfire_view)
            { return output_instance[o]->move_vp_vertical(+1); });
        activate_   .set_handler([=](wf::output_t *o, wayfire_view)
            { return output_instance[o]->input_grabbed(); });
    }
    void fini() override { this->fini_output_tracking(); }
};

DECLARE_WAYFIRE_PLUGIN(WayfireVerticalExpo);
