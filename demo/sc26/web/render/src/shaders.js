// GLSL for the three ways of drawing a protein and the light around them.

// Points of light with depth of field. Each atom is a sprite whose blur circle grows with its
// distance from the focal plane (thin-lens circle of confusion) while its total energy stays
// constant, so far noise reads as soft bokeh and the folded core reads as crisp sparks.
export const POINTS_VS = `
layout(location=0) in vec3 aA;
layout(location=1) in vec3 aB;
layout(location=2) in vec3 aColor;
uniform mat4 uView, uProj;
uniform float uAlpha, uProjScale, uFocus, uAperture, uSlab, uRadius, uMaxPx, uGain, uNear;
out vec3 vColor;
out float vSharp;
const float HALO = 2.2;
void main() {
  vec4 v = uView * vec4(mix(aA, aB, uAlpha), 1.0);
  float d = max(-v.z, 1e-3);
  float rpx = uRadius * uProjScale / d;
  float coc = uAperture * max(0.0, abs(d - uFocus) - uSlab) / d * uProjScale;  // the protein itself stays sharp
  float s = sqrt(rpx * rpx + coc * coc);
  float sc = min(s, uMaxPx);
  float energy = (rpx * rpx) / (sc * sc + 1e-6) * min(1.0, sc / s);
  vColor = aColor * uGain * energy * smoothstep(uNear, 2.5 * uNear, d);
  vSharp = rpx / s;
  gl_PointSize = 2.0 * HALO * max(sc, 0.75);
  gl_Position = uProj * v;
}`;

export const POINTS_FS = `
in vec3 vColor;
in float vSharp;
out vec4 o;
const float HALO = 2.2;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0) * HALO;
  if (r > HALO) discard;
  float spark = exp(-r * r * 2.2) + 0.05 * exp(-r * r * 0.35);
  float bokeh = smoothstep(1.0, 0.82, r) * (0.75 + 0.25 * r * r) * 0.42;
  o = vec4(vColor * mix(bokeh, spark, vSharp * vSharp), 0.0);
}`;

// Lit surface. The light rig is fixed to the camera, so orbiting never walks the protein into
// shadow: a warm key from upper left, a cool fill, a sky/ground ambient, a satin highlight, and a
// cool rim that separates the silhouette from the dark ground. Occlusion comes from the mesher.
// uSolid blends from glass (rim only, see-through face-on) to an opaque skin.
export const SURFACE_VS = `
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec4 aCol;
uniform mat4 uView, uProj;
out vec3 vN, vP, vCol;
out float vAO;
void main() {
  vec4 v = uView * vec4(aPos, 1.0);
  vP = v.xyz;
  vN = mat3(uView) * aNrm;
  vCol = aCol.rgb * aCol.rgb;  // stored gamma-ish, light in linear
  vAO = aCol.a;
  gl_Position = uProj * v;
}`;

export const SURFACE_FS = `
in vec3 vN, vP, vCol;
in float vAO;
uniform float uOpacity, uSolid, uEmissive;
out vec4 o;
const vec3 KEY = normalize(vec3(-0.55, 0.65, 0.55));
const vec3 FILL = normalize(vec3(0.7, -0.2, 0.45));
void main() {
  vec3 N = normalize(vN), V = normalize(-vP);
  if (dot(N, V) < 0.0) N = -N;
  float nk = dot(N, KEY);
  float ao = vAO * vAO;  // the mesher's openness, squared: crevices go properly dark
  vec3 key = vCol * max(0.0, (nk + 0.2) / 1.2) * vec3(1.0, 0.95, 0.88) * 0.95;
  vec3 fill = vCol * max(0.0, dot(N, FILL)) * vec3(0.30, 0.42, 0.70) * 0.30;
  float hemi = 0.5 + 0.5 * N.y;
  vec3 amb = vCol * mix(vec3(0.015, 0.02, 0.04), vec3(0.10, 0.12, 0.17), hemi);
  vec3 H = normalize(KEY + V);
  float nh = max(0.0, dot(N, H));
  float sheen = (pow(nh, 24.0) * 0.22 + pow(nh, 180.0) * 0.9) * smoothstep(-0.1, 0.3, nk);  // satin + clearcoat
  float nv = max(0.0, dot(N, V));
  float fres = pow(1.0 - nv, 4.0);
  vec3 rim = mix(vec3(0.35, 0.60, 1.0), vCol * 1.6, 0.35) * fres * 0.75;
  vec3 c = (key + fill + amb) * ao + vec3(sheen) * vAO + rim * (0.25 + 0.75 * vAO) + vCol * uEmissive;
  float a = mix(clamp(0.05 + 1.2 * pow(1.0 - nv, 3.0), 0.0, 1.0), 1.0, uSolid) * uOpacity;
  o = vec4(c * a, a);
}`;

// Bloom: 13-tap downsample (Jimenez 2014) and a 9-tap tent upsample, added level by level.
export const DOWN_FS = `
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uKnee;  // > 0 on the first level only: what glows is light above ~1, i.e. the points
out vec4 o;
vec3 t(vec2 d) {
  vec3 c = texture(uSrc, vUv + d * uTexel).rgb;
  if (uKnee > 0.0) { float l = max(c.r, max(c.g, c.b)); c *= smoothstep(uKnee * 0.5, uKnee * 1.5, l); }
  return c;
}
void main() {
  vec3 a = t(vec2(-2, 2)), b = t(vec2(0, 2)), c = t(vec2(2, 2));
  vec3 d = t(vec2(-2, 0)), e = t(vec2(0, 0)), f = t(vec2(2, 0));
  vec3 g = t(vec2(-2, -2)), h = t(vec2(0, -2)), i = t(vec2(2, -2));
  vec3 j = t(vec2(-1, 1)), k = t(vec2(1, 1)), l = t(vec2(-1, -1)), m = t(vec2(1, -1));
  vec3 s = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  o = vec4(s, 1.0);
}`;

export const UP_FS = `
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uTexel;
out vec4 o;
vec3 t(vec2 d) { return texture(uSrc, vUv + d * uTexel).rgb; }
void main() {
  vec3 s = t(vec2(0, 0)) * 4.0 + (t(vec2(-1, 0)) + t(vec2(1, 0)) + t(vec2(0, -1)) + t(vec2(0, 1))) * 2.0
         + t(vec2(-1, -1)) + t(vec2(1, -1)) + t(vec2(-1, 1)) + t(vec2(1, 1));
  o = vec4(s / 16.0, 1.0);
}`;

// The ground: a near-black blue with a faint lift behind the protein. Drawn first into the scene
// target, so the scene needs no alpha channel and can live in a 32-bit HDR format.
export const BG_FS = `
in vec2 vUv;
uniform vec2 uRes, uCenter;
out vec4 o;
void main() {
  float r = length((vUv - uCenter) * vec2(uRes.x / uRes.y, 1.0));
  o = vec4(mix(vec3(0.010, 0.014, 0.030), vec3(0.0015, 0.002, 0.004), smoothstep(0.0, 1.0, r)), 1.0);
}`;

// Tone and finish: ACES-fitted curve, a soft vignette, and a 1/255 dither so the dark gradient
// never bands on a big panel.
export const COMPOSITE_FS = `
in vec2 vUv;
uniform sampler2D uScene, uBloom;
uniform vec2 uRes;
uniform float uBloomK, uExposure, uSeed;
out vec4 o;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031 + uSeed); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
void main() {
  vec3 c = (texture(uScene, vUv).rgb + texture(uBloom, vUv).rgb * uBloomK) * uExposure;
  c = srgb(aces(c));
  vec2 e = (vUv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
  c *= mix(1.0, 0.72, smoothstep(0.45, 1.15, length(e)));
  c += (hash(gl_FragCoord.xy) + hash(gl_FragCoord.yx + 7.0) - 1.0) / 255.0;
  o = vec4(c, 1.0);
}`;
