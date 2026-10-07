// Oriented surfels rendered into FMCW range bins: profiles, JVP, VJP and dictionary rows.
// The Dirichlet kernel is evaluated in cotangent form; the scattering is scattering.slang.
//
// With N FFT bins, S ADC samples, padding P = N / S and d = bin - k0, the
// unit-peak negative-exponent kernel used throughout this library is
//
//   D(d) = sin(pi d / P) / (S sin(pi d / N)) * exp(-i pi (S - 1) d / N)
//        = nu(d) / S * (cot(pi d / N) + i),   nu(d) = sin(pi d / P) exp(-i pi d / P),
//   D'(d) = ((pi / P) q(d) (cot + i) - (pi / N) nu(d) (1 + cot^2)) / S,
//   q(d) = exp(-2i pi d / P),  nu = (1 - q) / (2i).
//
// This is the same kernel evaluated at every observed bin. q and nu have period
// P bins, so one surfel needs P numerator values per aperture, and each bin costs
// one cotangent obtained by exact angle addition from one anchor per tile.
// Numerator and cotangent both degenerate at the bin nearest to k0; that single
// bin uses the directly evaluated kernel instead. There is no near-lobe branch.
//
// Compile-time parameters: PAD is the padding factor P, TILE the number of
// consecutive bins sharing one anchor. TILE must be a multiple of PAD.
#ifndef PAD
#define PAD 4
#endif
#ifndef TILE
#define TILE 16
#endif

#include "scattering.slang"
// Numerator values of one aperture-surfel pair for every bin class r = bin mod PAD.
struct BinClasses
{
    float2 q[PAD];   // exp(-2i pi (r - k0) / PAD)
    float2 nu[PAD];  // (1 - q) / (2i)
    int lobe;        // bin nearest to k0, wrapped into [0, n_fft)
    float offset;    // lobe - k0 before wrapping, in [-0.5, 0.5]
    float sp;        // sin and cos of pi * offset / PAD
    float cp;
};

// k0 lies in [0, n_fft]; turn = exp(-2i pi / PAD) advances q by one bin.
BinClasses bin_classes(float k0, int n_fft, float pad_angle, float2 turn)
{
    BinClasses c;
    float nearest=round(k0);
    int lobe=int(nearest);
    c.offset=nearest-k0;
    c.lobe=lobe==n_fft ? 0 : lobe;
    // The small offset and the integer residue are rotated separately so that
    // the numerator does not inherit the rounding of a large phase.
    float sr,cr;
    sincos(pad_angle*c.offset,c.sp,c.cp);
    sincos(2.0f*pad_angle*float(lobe%PAD),sr,cr);
    float2 q=cmul(float2(c.cp*c.cp-c.sp*c.sp,-2.0f*c.sp*c.cp),float2(cr,sr));
    for(int r=0;r<PAD;r++) {
        c.q[r]=q;
        c.nu[r]=float2(-0.5f*q.y,-0.5f*(1.0f-q.x));
        q=cmul(q,turn);
    }
    return c;
}

// Per aperture-surfel pair: amplitude times carrier phase, and the peak bin k0
// wrapped into [0, n_fft). Back-facing pairs store a zero amplitude.
[AutoPyBindCUDA] [CUDAKernel]
void pair_geometry(TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> amplitude, TensorView<float> peak,
    int num_apertures, int num_surfels, int n_fft, float k0_per_meter, float fc,
    float slope, float t_start, float lambda_m, float fresnel_n)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=num_apertures*num_surfels) return;
    int a=i/num_surfels, s=i%num_surfels;
    float dx=scan_positions[a,0]-centers[s,0];
    float dy=scan_positions[a,1]-centers[s,1];
    float dz=scan_positions[a,2]-centers[s,2];
    float dist=sqrt(dx*dx+dy*dy+dz*dz)+1e-12f;
    float inv_dist=1.0f/dist;
    float cos_angle=(dx*normals[s,0]+dy*normals[s,1]+dz*normals[s,2])*inv_dist;
    float amp,amp_cos;
    scattering(cos_angle,dist,fresnel_n,lambda_m,amp,amp_cos);
    float2 c=amp*carrier(dist,fc,slope,t_start);
    float bin=dist*k0_per_meter;
    bin-=float(n_fft)*floor(bin/float(n_fft));
    amplitude[a,s,0]=c.x; amplitude[a,s,1]=c.y;
    peak[a,s]=bin;
}

// output[a, part, k] = sum over one segment of surfels of w c D(k - k0).
// Grid: x = TILE-bin tiles, y = apertures, z = surfel segments. The host sums
// the segments, so no atomics are needed and the order of addition is fixed.
[AutoPyBindCUDA] [CUDAKernel]
void render_bins(TensorView<float> amplitude, TensorView<float> peak, TensorView<float> weights,
    TensorView<float> output, int num_apertures, int num_surfels, int num_bins,
    int n_fft, int samples, int segment)
{
    int tile=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    int a=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    int part=cudaBlockIdx().z;
    int first=tile*TILE;
    if(first>=num_bins || a>=num_apertures) return;
    float bin_angle=PI/float(n_fft), pad_angle=PI/float(PAD);
    float inv_samples=1.0f/float(samples);
    float anchor=float(first+TILE/2);
    // Angle-addition constants of every bin relative to the anchor bin.
    float cj[TILE];
    float sj[TILE];
    float2 acc[TILE];
    for(int j=0;j<TILE;j++) {
        sincos(bin_angle*float(j-TILE/2),sj[j],cj[j]);
        acc[j]=float2(0,0);
    }
    float2 turn=cis(-2.0f*pad_angle);
    int stop=min(num_surfels,(part+1)*segment);
    for(int s=part*segment;s<stop;s++) {
        float2 c=float2(amplitude[a,s,0],amplitude[a,s,1]);
        if(c.x==0.0f && c.y==0.0f) continue;
        float k0=peak[a,s];
        BinClasses classes=bin_classes(k0,n_fft,pad_angle,turn);
        float2 cw=cmul(c,float2(weights[s,0],weights[s,1]))*inv_samples;
        float2 g[PAD];
        for(int r=0;r<PAD;r++) g[r]=cmul(cw,classes.nu[r]);
        // Directly evaluated kernel at the bin nearest to k0.
        float sb,cb;
        sincos(pad_angle*classes.offset*inv_samples,sb,cb);
        float ratio=classes.offset==0.0f ? 1.0f : classes.sp/(float(samples)*sb);
        float2 nearest_term=cmul(cw,
            ratio*cmul(float2(classes.cp,-classes.sp),float2(cb,sb)))*float(samples);
        float s0,c0;
        sincos(bin_angle*(anchor-k0),s0,c0);
        for(int j=0;j<TILE;j++) {
            float sn=s0*cj[j]+c0*sj[j];
            float cs=c0*cj[j]-s0*sj[j];
            float cot=cs/sn;
            float2 gr=g[j%PAD];
            float2 term=float2(gr.x*cot-gr.y,gr.y*cot+gr.x);
            if(first+j==classes.lobe) term=nearest_term;
            acc[j]+=term;
        }
    }
    for(int j=0;j<TILE;j++) {
        if(first+j<num_bins) { output[a,part,first+j,0]=acc[j].x; output[a,part,first+j,1]=acc[j].y; }
    }
}

// Directional derivative of the rendered profiles for geometry and weight tangents.
// Same grid and segment sums as render_bins; geometry is evaluated per tile.
[AutoPyBindCUDA] [CUDAKernel]
void jvp_bins(TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> weights,
    TensorView<float> delta_centers, TensorView<float> delta_normals,
    TensorView<float> delta_weights, TensorView<float> output,
    int num_apertures, int num_surfels, int num_bins, int n_fft, int samples,
    float k0_per_meter, float fc, float slope, float t_start, float lambda_m,
    float fresnel_n, int segment)
{
    int tile=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    int a=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    int part=cudaBlockIdx().z;
    int first=tile*TILE;
    if(first>=num_bins || a>=num_apertures) return;
    float bin_angle=PI/float(n_fft), pad_angle=PI/float(PAD);
    float inv_samples=1.0f/float(samples);
    float anchor=float(first+TILE/2);
    float cj[TILE];
    float sj[TILE];
    float2 acc[TILE];
    for(int j=0;j<TILE;j++) {
        sincos(bin_angle*float(j-TILE/2),sj[j],cj[j]);
        acc[j]=float2(0,0);
    }
    float2 turn=cis(-2.0f*pad_angle);
    float3 sensor=float3(scan_positions[a,0],scan_positions[a,1],scan_positions[a,2]);
    int stop=min(num_surfels,(part+1)*segment);
    for(int s=part*segment;s<stop;s++) {
        float3 n=float3(normals[s,0],normals[s,1],normals[s,2]);
        float3 dc=float3(delta_centers[s,0],delta_centers[s,1],delta_centers[s,2]);
        float3 dn=float3(delta_normals[s,0],delta_normals[s,1],delta_normals[s,2]);
        float3 d=sensor-float3(centers[s,0],centers[s,1],centers[s,2]);
        float r=length(d)+1e-12f;
        float3 u=d/r;
        float cosine=dot(u,n);
        if(cosine<=0.0f) continue;
        float dr=-dot(u,dc);
        float dcos=dot((cosine*u-n)/r,dc)+dot(u,dn);
        float amp,amp_cos;
        scattering(cosine,r,fresnel_n,lambda_m,amp,amp_cos);
        float damp=-amp*dr/r+amp_cos*dcos;
        float2 rot=carrier(r,fc,slope,t_start);
        float dphase=carrier_rate(r,fc,slope,t_start)*dr;
        float2 w=float2(weights[s,0],weights[s,1]);
        float2 dw=float2(delta_weights[s,0],delta_weights[s,1]);
        // Contribution = e1 D(d) + e2 D'(d).
        float2 e1=cmul(rot,cmul(float2(damp,amp*dphase),w)+amp*dw);
        float2 e2=cmul(rot,w)*(-amp*k0_per_meter*dr);
        float k0=r*k0_per_meter;
        k0-=float(n_fft)*floor(k0/float(n_fft));
        BinClasses classes=bin_classes(k0,n_fft,pad_angle,turn);
        float2 h1[PAD];
        float2 h2[PAD];
        for(int p=0;p<PAD;p++) {
            h1[p]=(cmul(e1,classes.nu[p])+pad_angle*cmul(e2,classes.q[p]))*inv_samples;
            h2[p]=cmul(e2,classes.nu[p])*(-bin_angle*inv_samples);
        }
        float2 value,slope_p;
        dirichlet(classes.offset/float(PAD),samples,samples,value,slope_p);
        float2 nearest_term=cmul(e1,value)+cmul(e2,slope_p)/float(PAD);
        float s0,c0;
        sincos(bin_angle*(anchor-k0),s0,c0);
        for(int j=0;j<TILE;j++) {
            float sn=s0*cj[j]+c0*sj[j];
            float cs=c0*cj[j]-s0*sj[j];
            float cot=cs/sn;
            float2 a1=h1[j%PAD];
            float2 term=float2(a1.x*cot-a1.y,a1.y*cot+a1.x)+h2[j%PAD]*(1.0f+cot*cot);
            if(first+j==classes.lobe) term=nearest_term;
            acc[j]+=term;
        }
    }
    for(int j=0;j<TILE;j++) {
        if(first+j<num_bins) { output[a,part,first+j,0]=acc[j].x; output[a,part,first+j,1]=acc[j].y; }
    }
}

// Adjoint of jvp_bins for one surfel and 32 apertures per warp. Each thread
// reduces all bins; the warp sums its apertures and writes one partial result
// output[s, block, 0..7] = (grad center xyz, grad normal xyz, grad weight re, im).
// The host adds the partial results, so there are no atomics.
[AutoPyBindCUDA] [CUDAKernel]
void vjp_bins(TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> weights, TensorView<float> upstream,
    TensorView<float> output, int num_apertures, int num_surfels, int num_bins,
    int n_fft, int samples, float k0_per_meter, float fc, float slope, float t_start,
    float lambda_m, float fresnel_n)
{
    uint s=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    uint source=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(s>=num_surfels) return; // whole warp; all lanes of valid surfels reduce
    uint a=min(source,uint(num_apertures-1));
    float bin_angle=PI/float(n_fft), pad_angle=PI/float(PAD);
    float inv_samples=1.0f/float(samples);
    float3 n=float3(normals[s,0],normals[s,1],normals[s,2]);
    float3 d=float3(scan_positions[a,0]-centers[s,0],scan_positions[a,1]-centers[s,1],
        scan_positions[a,2]-centers[s,2]);
    float r=length(d)+1e-12f;
    float3 u=d/r;
    float cosine=dot(u,n);
    bool visible=cosine>0.0f && source<num_apertures;
    float3 gc=float3(0,0,0), gn=float3(0,0,0);
    float2 gb=float2(0,0);
    if(visible) {
        float amp,amp_cos;
        scattering(cosine,r,fresnel_n,lambda_m,amp,amp_cos);
        float2 rot=carrier(r,fc,slope,t_start);
        float phase_r=carrier_rate(r,fc,slope,t_start);
        float k0=r*k0_per_meter;
        k0-=float(n_fft)*floor(k0/float(n_fft));
        BinClasses classes=bin_classes(k0,n_fft,pad_angle,cis(-2.0f*pad_angle));
        float cj[TILE];
        float sj[TILE];
        for(int j=0;j<TILE;j++) sincos(bin_angle*float(j-TILE/2),sj[j],cj[j]);
        // Class sums of (cot - i) g and (1 + cot^2) g; the nearest bin is kept apart.
        float2 first_sum[PAD];
        float2 second_sum[PAD];
        for(int p=0;p<PAD;p++) { first_sum[p]=float2(0,0); second_sum[p]=float2(0,0); }
        float2 nearest_g=float2(0,0);
        for(int first=0;first<num_bins;first+=TILE) {
            float s0,c0;
            sincos(bin_angle*(float(first+TILE/2)-k0),s0,c0);
            for(int j=0;j<TILE;j++) {
                int k=first+j;
                if(k<num_bins) {
                    float2 g=float2(upstream[a,k,0],upstream[a,k,1]);
                    float sn=s0*cj[j]+c0*sj[j];
                    float cs=c0*cj[j]-s0*sj[j];
                    float cot=cs/sn;
                    if(k==classes.lobe) nearest_g=g;
                    else {
                        first_sum[j%PAD]+=float2(g.x*cot+g.y,g.y*cot-g.x);
                        second_sum[j%PAD]+=g*(1.0f+cot*cot);
                    }
                }
            }
        }
        float2 value,slope_p;
        dirichlet(classes.offset/float(PAD),samples,samples,value,slope_p);
        // g0 = sum conj(D) g and g1 = sum conj(D') g over all bins.
        float2 g0=cmul(conj(value),nearest_g);
        float2 g1=cmul(conj(slope_p),nearest_g)/float(PAD);
        for(int p=0;p<PAD;p++) {
            float2 by_nu=cmul(conj(classes.nu[p]),first_sum[p]);
            g0+=by_nu*inv_samples;
            g1+=(pad_angle*cmul(conj(classes.q[p]),first_sum[p])
                -bin_angle*cmul(conj(classes.nu[p]),second_sum[p]))*inv_samples;
        }
        float2 w=float2(weights[s,0],weights[s,1]);
        float2 z0=cmul(conj(rot),g0), z1=cmul(conj(rot),g1);
        float2 cw=cmul(conj(w),z0);
        float along=cmul(conj(w),z1).x*k0_per_meter;
        float gr=-cw.x*amp/r+amp*(cw.y*phase_r-along);
        float gcos=cw.x*amp_cos;
        gc=-gr*u+gcos*(cosine*u-n)/r;
        gn=gcos*u;
        gb=amp*z0;
    }
    float total[8];
    for(int j=0;j<3;j++) { total[j]=WaveActiveSum(gc[j]); total[3+j]=WaveActiveSum(gn[j]); }
    total[6]=WaveActiveSum(gb.x);
    total[7]=WaveActiveSum(gb.y);
    if(cudaThreadIdx().x==0) {
        for(int j=0;j<8;j++) output[s,cudaBlockIdx().x,j]=total[j];
    }
}

// Dictionary entries c sqrt(area) D(k - k0). A thread owns one surfel and one aperture
// and writes all bins; threads span adjacent surfels, so writes are coalesced.
// Real rows are (2 k + part) * num_apertures + a, so one product projects them; complex
// entries are output[a num_bins + k, 2 s + part]. part is 0 for real, 1 for imaginary.
[AutoPyBindCUDA] [CUDAKernel]
void design_bins(TensorView<float> scan_positions, TensorView<float> centers,
    TensorView<float> normals, TensorView<float> areas, TensorView<float> output,
    int num_apertures, int num_surfels, int num_bins, int n_fft, int samples,
    float k0_per_meter, float fc, float slope, float t_start, float lambda_m,
    float fresnel_n, bool complex_entries)
{
    int s=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    int a=cudaBlockIdx().y*cudaBlockDim().y+cudaThreadIdx().y;
    if(s>=num_surfels || a>=num_apertures) return;
    float bin_angle=PI/float(n_fft), pad_angle=PI/float(PAD);
    float dx=scan_positions[a,0]-centers[s,0];
    float dy=scan_positions[a,1]-centers[s,1];
    float dz=scan_positions[a,2]-centers[s,2];
    float dist=sqrt(dx*dx+dy*dy+dz*dz)+1e-12f;
    float cos_angle=(dx*normals[s,0]+dy*normals[s,1]+dz*normals[s,2])/dist;
    bool visible=cos_angle>0.0f;
    float2 g[PAD];
    float cj[TILE];
    float sj[TILE];
    float2 nearest_term=float2(0,0);
    float k0=0.0f;
    int lobe=-1;
    if(visible) {
        float amp,amp_cos;
        scattering(cos_angle,dist,fresnel_n,lambda_m,amp,amp_cos);
        float2 cw=amp*sqrt(areas[s])/float(samples)*carrier(dist,fc,slope,t_start);
        k0=dist*k0_per_meter;
        k0-=float(n_fft)*floor(k0/float(n_fft));
        BinClasses classes=bin_classes(k0,n_fft,pad_angle,cis(-2.0f*pad_angle));
        for(int r=0;r<PAD;r++) g[r]=cmul(cw,classes.nu[r]);
        float sb,cb;
        sincos(pad_angle*classes.offset/float(samples),sb,cb);
        float ratio=classes.offset==0.0f ? 1.0f : classes.sp/(float(samples)*sb);
        nearest_term=cmul(cw,
            ratio*cmul(float2(classes.cp,-classes.sp),float2(cb,sb)))*float(samples);
        for(int j=0;j<TILE;j++) sincos(bin_angle*float(j-TILE/2),sj[j],cj[j]);
        lobe=classes.lobe;
    }
    for(int first=0;first<num_bins;first+=TILE) {
        float s0=1.0f, c0=0.0f;
        if(visible) sincos(bin_angle*(float(first+TILE/2)-k0),s0,c0);
        for(int j=0;j<TILE;j++) {
            int k=first+j;
            if(k<num_bins) {
                float2 term=float2(0,0);
                if(visible) {
                    float cot=(c0*cj[j]-s0*sj[j])/(s0*cj[j]+c0*sj[j]);
                    float2 gr=g[j%PAD];
                    term=k==lobe ? nearest_term : float2(gr.x*cot-gr.y,gr.y*cot+gr.x);
                }
                if(complex_entries) {
                    output[a*num_bins+k,2*s]=term.x;
                    output[a*num_bins+k,2*s+1]=term.y;
                } else {
                    output[2*k*num_apertures+a,s]=term.x;
                    output[(2*k+1)*num_apertures+a,s]=term.y;
                }
            }
        }
    }
}
