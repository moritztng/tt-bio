// GLSL for the two ways of drawing a protein: plain points while it folds, a lit mesh (the cartoon,
// or optionally a molecular surface) once it is done.
//
// One light model for everything, after the defaults of ChimeraX and Mol*: a matte material, one
// key light from the upper left fixed to the camera, a soft sky/ground ambient, ambient occlusion
// where the geometry provides it, and depth cueing (fog toward the ground colour). No bloom, glow,
// rim, glass or depth-of-field.

const LIGHT = `
const vec3 KEY = normalize(vec3(-0.45, 0.6, 0.66));
uniform vec3 uGround;
uniform vec2 uFog;   // view depth where fog starts, and where it is full
vec3 shade(vec3 col, vec3 N, float ao, float depth) {
  float key = max(0.0, dot(N, KEY));
  float sky = 0.5 + 0.5 * N.y;
  vec3 c = col * (0.85 * key * (0.55 + 0.45 * ao) + mix(0.22, 0.42, sky) * ao);
  c += vec3(0.06) * pow(max(0.0, dot(N, normalize(KEY + vec3(0.0, 0.0, 1.0)))), 24.0) * ao;
  return mix(c, uGround, 0.82 * smoothstep(uFog.x, uFog.y, depth));
}`;

// Each atom is a small matte sphere: a screen-facing disc shaded as a ball. Size is the atom's
// world radius under perspective, so depth reads as size, capped so a noise atom passing close to
// the camera never becomes a blob. Opaque and depth-tested; edges antialiased by alpha-to-coverage.
export const POINTS_VS = `
layout(location=0) in vec3 aA;
layout(location=1) in vec3 aB;
layout(location=2) in vec3 aColor;
layout(location=3) in float aBall;   // 1 for atoms drawn as balls after the fold (ligands)
uniform mat4 uView, uProj;
uniform float uAlpha, uProjScale, uRadius, uBallRadius, uFold, uMinPx, uMaxPx, uNear;
out vec3 vColor;
out float vDepth, vPx;
void main() {
  vec4 v = uView * vec4(mix(aA, aB, uAlpha), 1.0);
  float d = max(-v.z, 1e-3);
  // folding: every atom a point; done: points shrink away into the cartoon, ligands grow to balls
  float point = clamp(uRadius * uProjScale / d, uMinPx, uMaxPx) * uFold;
  float ball = uBallRadius * uProjScale / d * (1.0 - uFold);
  vPx = mix(point, max(point, ball), aBall) * smoothstep(uNear, 2.0 * uNear, d);
  vColor = aColor;
  vDepth = d;
  gl_PointSize = 2.0 * vPx + 2.0;
  gl_Position = vPx > 0.05 ? uProj * v : vec4(2.0, 2.0, 2.0, 1.0);
}`;

export const POINTS_FS = `
in vec3 vColor;
in float vDepth, vPx;
out vec4 o;
${LIGHT}
void main() {
  vec2 q = (gl_PointCoord * 2.0 - 1.0) * (vPx + 1.0);   // pixels from the centre
  float cover = clamp(vPx - length(q) + 0.5, 0.0, 1.0);
  if (cover <= 0.0) discard;
  vec2 p = q / vPx;
  vec3 N = vec3(p.x, -p.y, sqrt(max(0.0, 1.0 - min(dot(p, p), 1.0))));
  o = vec4(shade(vColor, N, 1.0, vDepth), cover);
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
