// Surfel candidates against a residual: correlations a^H r and norms |a|^2, the
// normal-bank scores, their explicit derivatives, and the explicit dictionary Jacobian.
// The Dirichlet kernel is evaluated directly per bin; the scattering is scattering.slang.

#include "scattering.slang"

struct Complex
{
    float re;
    float im;
};

Complex cmul(Complex a, Complex b)
{
    return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// Reuse the same normalized PSF as the spectral renderer. x is in radians.
Complex dirichlet_complex(float x, float n)
{
    int samples=int(2.0f*n+1.0f);
    float2 v,d;
    dirichlet(x*float(samples)/TWO_PI,samples,samples,v,d);
    return Complex(v.x,v.y);
}
Complex dirichlet_complex_grad(float x, float n)
{
    int samples=int(2.0f*n+1.0f);
    float2 v,d;
    dirichlet(x*float(samples)/TWO_PI,samples,samples,v,d);
    d*=float(samples)/TWO_PI;
    return Complex(d.x,d.y);
}

struct CandGeometry
{
    bool visible;
    float amp;
    float k0;
    Complex rot;
};

CandGeometry cand_geometry(
    float px, float py, float pz,
    float cx, float cy, float cz,
    float nx, float ny, float nz,
    float k0_per_meter, float fc, float slope, float t_start, float lambda_m,
    float fresnel_n
)
{
    CandGeometry geo;
    float dx = px - cx;
    float dy = py - cy;
    float dz = pz - cz;
    float dist = sqrt(dx * dx + dy * dy + dz * dz) + 1e-12f;
    float inv_dist = 1.0f / dist;
    float cos_angle = (dx * nx + dy * ny + dz * nz) * inv_dist;
    if (cos_angle <= 0.0f)
    {
        geo.visible = false;
        geo.amp = 0.0f;
        geo.k0 = 0.0f;
        geo.rot = Complex(0.0f, 0.0f);
        return geo;
    }
    float amp_cos;
    scattering(cos_angle, dist, fresnel_n, lambda_m, geo.amp, amp_cos);
    float2 rot = carrier(dist, fc, slope, t_start);
    geo.visible = true;
    geo.k0 = dist * k0_per_meter;
    geo.rot = Complex(rot.x, rot.y);
    return geo;
}

// Candidate correlation: for candidate c and aperture a, accumulate
// conj(d_c) . residual and |d_c|^2 over bins, atomically per candidate.
// Grid: x = candidates, y = apertures.
[AutoPyBindCUDA] [CUDAKernel]
void surfel_correlate(TensorView<float> scan_positions, TensorView<float> candidates,
    TensorView<float> cand_normals, TensorView<float> residual_re, TensorView<float> residual_im,
    TensorView<float> corr_re, TensorView<float> corr_im, TensorView<float> norm2,
    int num_apertures, int num_candidates, int num_bins, int n_fft, float half_order,
    float k0_per_meter, float fc, float slope, float t_start, float lambda_m,
    float fresnel_n)
{
    uint cand=cudaThreadIdx().x+cudaBlockIdx().x*cudaBlockDim().x;
    uint aperture=cudaThreadIdx().y+cudaBlockIdx().y*cudaBlockDim().y;
    if(cand>=num_candidates || aperture>=num_apertures) return;
    CandGeometry geo=cand_geometry(
        scan_positions[aperture,0],scan_positions[aperture,1],scan_positions[aperture,2],
        candidates[cand,0],candidates[cand,1],candidates[cand,2],
        cand_normals[cand,0],cand_normals[cand,1],cand_normals[cand,2],
        k0_per_meter,fc,slope,t_start,lambda_m,fresnel_n);
    if(!geo.visible) return;
    float cr=0, ci=0, nn=0;
    for(int k=0;k<num_bins;k++) {
        Complex column=cmul(dirichlet_complex(TWO_PI*(float(k)-geo.k0)/float(n_fft),half_order),geo.rot);
        column.re*=geo.amp; column.im*=geo.amp;
        float re=residual_re[aperture,k], im=residual_im[aperture,k];
        cr+=column.re*re+column.im*im;
        ci+=column.re*im-column.im*re;
        nn+=column.re*column.re+column.im*column.im;
    }
    float old;
    corr_re.InterlockedAdd(cand,cr,old);
    corr_im.InterlockedAdd(cand,ci,old);
    norm2.InterlockedAdd(cand,nn,old);
}

// Score a normal bank while evaluating distance, phase and Dirichlet kernel only once
// per position and aperture.
[AutoPyBindCUDA] [CUDAKernel]
void surfel_correlate_orientations(
    TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> bank,
    TensorView<float> residual_re, TensorView<float> residual_im,
    TensorView<float> output, int num_apertures, int num_surfels, int num_normals,
    int num_bins, int n_fft, float half_order, float k0_per_meter, float fc,
    float slope, float t_start, float lambda_m, float fresnel_n,
    bool use_table, TensorView<float> range_table, TensorView<float> norm_table, int oversample)
{
    uint s=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    uint source=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(s>=num_surfels) return;
    uint a=min(source,uint(num_apertures-1));
    float3 d=float3(scan_positions[a,0]-centers[s,0],scan_positions[a,1]-centers[s,1],
        scan_positions[a,2]-centers[s,2]);
    float r=length(d)+1e-12f;
    float3 u=d/r;
    float2 rc=carrier(r,fc,slope,t_start);
    Complex rot=Complex(rc.x,rc.y);
    float cr=0.0f, ci=0.0f, norm=0.0f;
    if(use_table) {
        float x=r*k0_per_meter*float(oversample), t=x-floor(x);
        int base=int(floor(x)), size=n_fft*oversample;
        float t2=t*t, t3=t2*t;
        float weights[4]={-.5f*t+t2-.5f*t3,1-2.5f*t2+1.5f*t3,
            .5f*t+2*t2-1.5f*t3,-.5f*t2+.5f*t3};
        float2 value=float2(0,0);
        for(int j=0;j<4;j++) {
            int k=((base+j-1)%size+size)%size;
            value+=weights[j]*float2(range_table[a,k,0],range_table[a,k,1]);
            norm+=weights[j]*norm_table[k];
        }
        value=cmul(value,float2(rot.re,-rot.im)); cr=value.x; ci=value.y;
        norm=max(norm,0.0f);
    } else for(int k=0;k<num_bins;k++) {
        float x=TWO_PI*(float(k)-r*k0_per_meter)/float(n_fft);
        Complex z=cmul(dirichlet_complex(x,half_order),rot);
        float re=residual_re[a,k], im=residual_im[a,k];
        cr+=z.re*re+z.im*im;
        ci+=z.re*im-z.im*re;
        norm+=z.re*z.re+z.im*z.im;
    }
    float old;
    for(int j=0;j<=num_normals;j++) {
        float3 n=j==0 ? float3(normals[s,0],normals[s,1],normals[s,2]) :
            float3(bank[j-1,0],bank[j-1,1],bank[j-1,2]);
        float amp, amp_cos;
        scattering(dot(u,n),r,fresnel_n,lambda_m,amp,amp_cos);
        if(source>=num_apertures) amp=0.0f;
        float re=WaveActiveSum(amp*cr), im=WaveActiveSum(amp*ci), nn=WaveActiveSum(amp*amp*norm);
        if(cudaThreadIdx().x==0) {
            output.InterlockedAdd(uint3(s,j,0),re,old);
            output.InterlockedAdd(uint3(s,j,1),im,old);
            output.InterlockedAdd(uint3(s,j,2),nn,old);
        }
    }
}

// Candidate correlation and its explicit xyz differential. No observation-by-
// candidate Jacobian is written. Outputs: Re(c), Im(c), norm2, Re(dc)[3],
// Im(dc)[3], d(norm2)[3]. Each thread reduces one aperture in registers.
[AutoPyBindCUDA] [CUDAKernel]
void surfel_correlate_gradient(
    TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> residual_re,
    TensorView<float> residual_im, TensorView<float> output,
    int num_apertures, int num_surfels, int num_bins, int n_fft,
    float half_order, float k0_per_meter, float fc, float slope,
    float t_start, float lambda_m, float fresnel_n, bool include_normals)
{
    uint s=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    uint source=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(s>=num_surfels) return; // whole warp; all lanes of valid surfels reduce
    uint a=min(source,uint(num_apertures-1));
    float3 d=float3(scan_positions[a,0]-centers[s,0],scan_positions[a,1]-centers[s,1],
        scan_positions[a,2]-centers[s,2]);
    float3 n=float3(normals[s,0],normals[s,1],normals[s,2]);
    float r=length(d)+1e-12f;
    float3 u=d/r;
    float cosine=dot(u,n);
    bool visible=cosine>0.0f && source<num_apertures;
    float amp, amp_cos;
    scattering(cosine,r,fresnel_n,lambda_m,amp,amp_cos);
    float2 rc=carrier(r,fc,slope,t_start);
    Complex rot=Complex(rc.x,rc.y);
    float phase_r=carrier_rate(r,fc,slope,t_start);
    int width=include_normals ? 6 : 3;
    // Geometry factors are constant over range bins. Reduce six sufficient
    // statistics first, then chain xyz/normal derivatives once per aperture.
    float2 corr=float2(0,0), corr_x=float2(0,0);
    float norm=0, norm_x=0;
    int samples=int(2.0f*half_order+1.0f);
    for(int k=0;k<(visible ? num_bins : 0);k++) {
        float x=TWO_PI*(float(k)-r*k0_per_meter)/float(n_fft);
        float2 kernel,deriv;
        dirichlet(x*float(samples)/TWO_PI,samples,samples,kernel,deriv);
        float2 rotation=float2(rot.re,rot.im);
        float2 z=cmul(kernel,rotation), dz=cmul(deriv*float(samples)/TWO_PI,rotation);
        float2 res=float2(residual_re[a,k],residual_im[a,k]);
        corr+=cmul(conj(z),res); corr_x+=cmul(conj(dz),res);
        norm+=dot(z,z); norm_x+=dot(dz,z);
    }
    float old;
    float cr=WaveActiveSum(amp*corr.x), ci=WaveActiveSum(amp*corr.y), nn=WaveActiveSum(amp*amp*norm);
    if(cudaThreadIdx().x==0) {
        output.InterlockedAdd(uint2(s,0),cr,old);
        output.InterlockedAdd(uint2(s,1),ci,old);
        output.InterlockedAdd(uint2(s,2),nn,old);
    }
    for(int j=0;j<width;j++) {
        float da=j<3 ? amp*u[j]/r+amp_cos*(cosine*u[j]-n[j])/r : amp_cos*u[j-3];
        float dx=j<3 ? TWO_PI*k0_per_meter/float(n_fft)*u[j] : 0;
        float dphase=j<3 ? phase_r*u[j] : 0;
        float2 dc=da*corr+amp*(dx*corr_x+dphase*float2(-corr.y,corr.x));
        float gx=WaveActiveSum(dc.x), gy=WaveActiveSum(dc.y);
        float gn=WaveActiveSum(2.0f*amp*(da*norm+amp*dx*norm_x));
        if(cudaThreadIdx().x==0) {
            output.InterlockedAdd(uint2(s,3+j),gx,old);
            output.InterlockedAdd(uint2(s,3+width+j),gy,old);
            output.InterlockedAdd(uint2(s,3+2*width+j),gn,old);
        }
    }
}

// Explicit position and normal Jacobian for generic surfel variable projection.
// The host applies parameter masks, units and unit-normal tangent projection.
[AutoPyBindCUDA] [CUDAKernel]
void surfel_design_jacobian(TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> output, TensorView<float> jacobian,
    int num_apertures, int num_surfels, int num_bins, int n_fft, float half_order,
    float k0_per_meter, float fc, float slope, float t_start, float lambda_m,
    float fresnel_n)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=num_apertures*num_bins*num_surfels) return;
    int s=i%num_surfels, k=(i/num_surfels)%num_bins, a=i/(num_surfels*num_bins);
    float3 d=float3(scan_positions[a,0]-centers[s,0],scan_positions[a,1]-centers[s,1],
        scan_positions[a,2]-centers[s,2]);
    float3 n=float3(normals[s,0],normals[s,1],normals[s,2]);
    float r=length(d)+1e-12f;
    float3 u=d/r;
    float cosine=dot(u,n);
    float2 value=float2(0,0);
    float2 derivative[6];
    for(int j=0;j<6;j++) derivative[j]=float2(0,0);
    if(cosine>0.0f) {
        float amp, amp_cos;
        scattering(cosine,r,fresnel_n,lambda_m,amp,amp_cos);
        float2 rc=carrier(r,fc,slope,t_start);
        Complex rot=Complex(rc.x,rc.y);
        float phase_r=carrier_rate(r,fc,slope,t_start);
        float x=TWO_PI*(float(k)-r*k0_per_meter)/float(n_fft);
        Complex kernel=cmul(dirichlet_complex(x,half_order),rot);
        Complex dk=cmul(dirichlet_complex_grad(x,half_order),rot);
        float2 z=float2(kernel.re,kernel.im), dz=float2(dk.re,dk.im);
        value=amp*z;
        for(int j=0;j<3;j++) {
            float da=amp*u[j]/r+amp_cos*(cosine*u[j]-n[j])/r;
            derivative[j]=da*z+amp*(dz*TWO_PI*k0_per_meter/float(n_fft)*u[j]
                -float2(-z.y,z.x)*phase_r*u[j]);
            derivative[j+3]=amp_cos*u[j]*z;
        }
    }
    output[a,k,s,0]=value.x; output[a,k,s,1]=value.y;
    for(int j=0;j<6;j++) {
        jacobian[a,k,s,j,0]=derivative[j].x;
        jacobian[a,k,s,j,1]=derivative[j].y;
    }
}
