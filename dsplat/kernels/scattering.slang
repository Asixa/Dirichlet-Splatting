// Scattering of an oriented surfel toward a monostatic sensor, shared by every surfel kernel.
//
// A surfel with center c and outward unit normal n, seen from a sensor at p:
//   r = |p - c|,  u = (p - c) / r,  cos = u . n
//   amplitude = cos^2 F(cos) lambda / (8 pi r)    zero for a back face, cos <= 0
//   F(cos) = (root - cos) / (root + cos),  root = sqrt(n_m^2 - 1 + cos^2)
// F is the s-polarized Fresnel reflection of a known dielectric with refractive index
// n_m; n_m <= 1 disables it. A dictionary column also carries sqrt(area), and the
// surfel's reflectivity is its coefficient. A linear chirp with carrier fc, slope K
// and ADC start t0 adds the phase 2 pi (fc tau + K tau (t0 - tau / 2)), tau = 2 r / c,
// and puts the echo at range bin r / spacing.

#include "common.slang"

// Fresnel factor F(cos) of refractive index n and its derivative with respect to cos.
void fresnel(float cosine, float n, out float value, out float grad)
{
    value = 1.0f;
    grad = 0.0f;
    if (n <= 1.0f) return;
    float root = sqrt(n * n - 1.0f + cosine * cosine);
    value = (root - cosine) / (root + cosine);
    grad = -2.0f * (root - cosine) / (root * (root + cosine));
}

// Amplitude cos^2 F(cos) lambda / (8 pi r) and its derivative with respect to cos.
void scattering(float cosine, float r, float fresnel_n, float lambda_m, out float amp, out float amp_cos)
{
    amp = 0.0f;
    amp_cos = 0.0f;
    if (cosine <= 0.0f) return;
    float f, df;
    fresnel(cosine, fresnel_n, f, df);
    float path = lambda_m / (8.0f * PI * r);
    amp = path * cosine * cosine * f;
    amp_cos = path * (2.0f * cosine * f + cosine * cosine * df);
}

// Chirp phase factor exp(2 pi i (fc tau + K tau (t0 - tau / 2))) of an echo at range r.
float2 carrier(float r, float fc, float slope, float t_start)
{
    float tau = 2.0f * r / C0;
    return cis(TWO_PI * (fc * tau + slope * tau * (t_start - 0.5f * tau)));
}

// Derivative of the chirp phase with respect to the range r.
float carrier_rate(float r, float fc, float slope, float t_start)
{
    return TWO_PI * 2.0f / C0 * (fc + slope * (t_start - 2.0f * r / C0));
}
