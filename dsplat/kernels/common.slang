// Complex algebra and the unit-peak, negative-exponent Dirichlet kernel D(d) with its
// derivative: D(d) = sum_n exp(-2 pi i n d / nfft) / samples, n < samples.
static const float PI = 3.14159265358979323846f;
static const float TWO_PI = 6.28318530717958647692f;
static const float C0 = 299792458.0f;  // vacuum wave speed, m/s
float2 cmul(float2 a, float2 b) { return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
float2 conj(float2 a) { return float2(a.x, -a.y); }
float2 cis(float x) { float s,c; sincos(x,s,c); return float2(c,s); }

void dirichlet(float delta, int samples, int nfft, out float2 value, out float2 deriv)
{
    float x = delta - float(nfft) * round(delta / float(nfft));
    float a = PI*float(samples)/float(nfft), b = PI/float(nfft);
    float h = PI*float(samples-1)/float(nfft);
    float ratio, dr;
    if (abs(x) < 1e-3f)
    {
        float c2 = (b*b-a*a)/6.0f;
        float c4 = (3*a*a*a*a+7*b*b*b*b-10*a*a*b*b)/360.0f;
        ratio = 1+c2*x*x+c4*x*x*x*x;
        dr = 2*c2*x+4*c4*x*x*x;
    }
    else
    {
        float sa,ca,sb,cb;
        sincos(a*x,sa,ca); sincos(b*x,sb,cb);
        ratio = sa/(float(samples)*sb);
        dr = (a*ca*sb-b*sa*cb)/(float(samples)*sb*sb);
    }
    float2 phase = cis(-h*x);
    value = ratio*phase;
    deriv = cmul(phase, float2(dr,-h*ratio));
}

void windowed(float delta, int samples, int nfft, int window, out float2 k, out float2 dk)
{
    dirichlet(delta, samples, nfft, k, dk);
    if (window == 0) return;
    float2 km,dm,kp,dp;
    float shift = float(nfft)/float(samples);
    dirichlet(delta-shift,samples,nfft,km,dm);
    dirichlet(delta+shift,samples,nfft,kp,dp);
    float w = window == 1 ? 0.5f : 0.46f/1.08f;
    k -= w*(km+kp); dk -= w*(dm+dp);
}
