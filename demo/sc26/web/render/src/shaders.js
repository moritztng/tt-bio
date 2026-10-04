// GLSL for the two ways of drawing a protein: plain points while it folds, a lit mesh (the cartoon,
// or optionally a molecular surface) once it is done.
//
// One light model for everything, after the defaults of PyMOL, ChimeraX and Mol*: a matte material,
// one key light from the upper left and a fill from the camera, both fixed to the camera, a soft
// sky/ground ambient, ambient occlusion where the geometry provides it, and depth cueing (fog toward
// the ground colour). No bloom, glow, rim, glass or depth-of-field.

const LIGHT = `
const vec3 KEY = normalize(vec3(-0.45, 0.6, 0.66));
uniform vec3 uGround;
uniform vec2 uFog;   // view depth where fog starts, and where it is full
vec3 fogged(vec3 c, float depth) { return mix(c, uGround, 0.65 * smoothstep(uFog.x, uFog.y, depth)); }
vec3 shade(vec3 col, vec3 N, float ao, float depth) {
  float key = max(0.0, dot(N, KEY));
  float sky = 0.5 + 0.5 * N.y;
  float head = max(0.0, N.z);   // a fill light from the camera, as PyMOL has: faces bright, edges dark
  vec3 c = col * (0.9 * key * (0.55 + 0.45 * ao) + 0.42 * head * (0.6 + 0.4 * ao) + mix(0.10, 0.22, sky) * ao);
  c += vec3(0.16) * pow(max(0.0, dot(N, normalize(KEY + vec3(0.0, 0.0, 1.0)))), 36.0) * ao;
  return fogged(c, depth);
}`;

// Each atom is a fine dot. Its size is set in screen pixels, not by the atom's world radius: at
// booth framing every atom would otherwise be a 6-20 px disc. Within [uMinPx, uMaxPx] the dot
// still shrinks with distance (uRef is the depth at which it is uMaxPx), so depth reads as size and
// fog, never as a blob. A dot under a pixel keeps its true area by fading rather than growing, so
// a dense cloud reads as texture. Flat colour, no shading: at two or three pixels a lit sphere is
// noise. Edges are analytic coverage, blended over.
export const POINTS_VS = `
layout(location=0) in vec3 aA;
layout(location=1) in vec3 aB;
layout(location=2) in vec3 aColor;
uniform mat4 uView, uProj;
uniform float uAlpha, uFold, uMinPx, uMaxPx, uRef, uNear;
out vec3 vColor;
out float vDepth, vPx, vInk;
void main() {
  vec4 v = uView * vec4(mix(aA, aB, uAlpha), 1.0);
  float d = max(-v.z, 1e-3);
  float r = clamp(uMaxPx * uRef / d, uMinPx, uMaxPx) * uFold * smoothstep(uNear, 2.0 * uNear, d);
  vPx = max(r, 0.75);
  vInk = (r * r) / (vPx * vPx);   // area of the true dot over the drawn one
  vColor = aColor;
  vDepth = d;
  gl_PointSize = 2.0 * vPx + 2.0;
  gl_Position = r > 0.02 ? uProj * v : vec4(2.0, 2.0, 2.0, 1.0);
}`;

export const POINTS_FS = `
in vec3 vColor;
in float vDepth, vPx, vInk;
out vec4 o;
${LIGHT}
void main() {
  float q = length(gl_PointCoord * 2.0 - 1.0) * (vPx + 1.0);   // pixels from the centre
  float a = clamp(vPx - q + 0.5, 0.0, 1.0) * vInk;
  if (a <= 0.004) discard;
  o = vec4(fogged(vColor, vDepth) * a, a);
}`;

export const MESH_VS = `
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec4 aCol;
uniform mat4 uView, uProj;
out vec3 vN, vCol;
out float vAO, vDepth;
void main() {
  vec4 v = uView * vec4(aPos, 1.0);
  vN = mat3(uView) * aNrm;
  vCol = aCol.rgb * aCol.rgb;   // stored as sqrt(linear) for 8-bit precision in the darks
  vAO = aCol.a;
  vDepth = -v.z;
  gl_Position = uProj * v;
}`;

export const MESH_FS = `
in vec3 vN, vCol;
in float vAO, vDepth;
uniform float uOpacity;
out vec4 o;
${LIGHT}
void main() {
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
  o = vec4(shade(vCol, N, vAO, vDepth) * uOpacity, uOpacity);
}`;

// The ground: one flat colour, the app's own.
export const BG_FS = `
uniform vec3 uGround;
out vec4 o;
void main() { o = vec4(uGround, 1.0); }`;

// Linear to sRGB, and a 1/255 dither so the dark ground never bands on a big panel.
export const COMPOSITE_FS = `
in vec2 vUv;
uniform sampler2D uScene;
uniform float uSeed;
out vec4 o;
vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031 + uSeed); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
void main() {
  vec3 c = srgb(clamp(texture(uScene, vUv).rgb, 0.0, 1.0));
  c += (hash(gl_FragCoord.xy) + hash(gl_FragCoord.yx + 7.0) - 1.0) / 255.0;
  o = vec4(c, 1.0);
}`;
