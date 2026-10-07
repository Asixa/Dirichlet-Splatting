#include "common.slang"

// All coordinates are bin units. At most three separable axes, no learned width.
void footprint(TensorView<float> centers, TensorView<float> bins,
    TensorView<int> samples, TensorView<int> nfft, TensorView<float> taps, float radius, int window, int dim,
    int s, int m, out float2 a, out float2 d0, out float2 d1, out float2 d2)
{
    float2 k[3], dk[3];
    for (int d=0;d<3;d++) { k[d]=float2(1,0); dk[d]=float2(0,0); }
    for (int d=0;d<dim;d++)
    {
        float delta=bins[m,d]-centers[s,d];
        delta-=float(nfft[d])*round(delta/float(nfft[d]));
        if(radius>0 && abs(delta)>radius) { k[d]=float2(0,0); dk[d]=float2(0,0); }
        else if(window==3)
        {
            k[d]=float2(0,0); dk[d]=float2(0,0);
            for(int n=0;n<samples[d];n++) {
                float omega=-2*PI*float(n)/float(nfft[d]);
                float2 term=taps[d,n]*cis(omega*delta);
                k[d]+=term; dk[d]+=cmul(float2(0,omega),term);
            }
        }
        else windowed(delta, samples[d], nfft[d], window, k[d], dk[d]);
    }
    a=cmul(cmul(k[0],k[1]),k[2]);
    d0=-cmul(cmul(dk[0],k[1]),k[2]);
    d1=-cmul(cmul(k[0],dk[1]),k[2]);
    d2=-cmul(cmul(k[0],k[1]),dk[2]);
}

[AutoPyBindCUDA] [CUDAKernel]
void spectral_forward(TensorView<float> centers, TensorView<float> coeff,
    TensorView<float> bins, TensorView<int> samples, TensorView<int> nfft, TensorView<float> taps, float radius,
    TensorView<float> output, int window, int dim, int count, int size)
{
    int m=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if (m>=size) return;
    float2 sum=float2(0,0);
    for (int s=0;s<count;s++)
    {
        float2 a,d0,d1,d2;
        footprint(centers,bins,samples,nfft,taps,radius,window,dim,s,m,a,d0,d1,d2);
        sum+=cmul(a,float2(coeff[s,0],coeff[s,1]));
    }
    output[m,0]=sum.x; output[m,1]=sum.y;
}

// Each (splat, bin tile) reduces locally then adds one parameter-gradient vector.
// PyTorch complex VJP convention: dL/dx = Re(conj(upstream) * dy/dx).
[AutoPyBindCUDA] [CUDAKernel]
void spectral_backward(TensorView<float> centers, TensorView<float> coeff,
    TensorView<float> bins, TensorView<int> samples, TensorView<int> nfft, TensorView<float> taps, float radius,
    TensorView<float> upstream, TensorView<float> grad_centers, TensorView<float> grad_coeff,
    int window, int dim, int count, int size, int span)
{
    int s=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if (s>=count) return;
    float3 gc=float3(0,0,0); float2 gb=float2(0,0);
    float2 b=float2(coeff[s,0],coeff[s,1]);
    int start=cudaBlockIdx().y*span;
    for (int m=start;m<min(start+span,size);m++)
    {
        float2 a,d0,d1,d2;
        footprint(centers,bins,samples,nfft,taps,radius,window,dim,s,m,a,d0,d1,d2);
        float2 g=float2(upstream[m,0],upstream[m,1]);
        gc+=float3(dot(g,cmul(b,d0)),dot(g,cmul(b,d1)),dot(g,cmul(b,d2)));
        gb+=cmul(conj(a),g);
    }
    float old;
    for(int d=0;d<dim;d++) grad_centers.InterlockedAdd(uint2(s,d),gc[d],old);
    grad_coeff.InterlockedAdd(uint2(s,0),gb.x,old);
    grad_coeff.InterlockedAdd(uint2(s,1),gb.y,old);
}

[AutoPyBindCUDA] [CUDAKernel]
void spectral_design(TensorView<float> centers, TensorView<float> bins,
    TensorView<int> samples, TensorView<int> nfft, TensorView<float> taps, float radius, TensorView<float> output,
    TensorView<float> jacobian, int window, int dim, int count, int size, bool derivatives)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=size*count) return;
    int m=i/count, s=i%count;
    float2 a,d0,d1,d2;
    footprint(centers,bins,samples,nfft,taps,radius,window,dim,s,m,a,d0,d1,d2);
    output[m,s,0]=a.x; output[m,s,1]=a.y;
    if (derivatives)
    {
        float2 ds[3]={d0,d1,d2};
        for(int d=0;d<dim;d++) { jacobian[m,s,d,0]=ds[d].x; jacobian[m,s,d,1]=ds[d].y; }
    }
}

// Directional derivative of the complete sum; never writes an M x S Jacobian.
[AutoPyBindCUDA] [CUDAKernel]
void spectral_jvp(TensorView<float> centers, TensorView<float> coeff,
    TensorView<float> delta_centers, TensorView<float> delta_coeff,
    TensorView<float> bins, TensorView<int> samples, TensorView<int> nfft,
    TensorView<float> taps, float radius, TensorView<float> output,
    int window, int dim, int count, int size)
{
    int m=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(m>=size) return;
    float2 sum=float2(0,0);
    for(int s=0;s<count;s++) {
        float2 a,d0,d1,d2;
        footprint(centers,bins,samples,nfft,taps,radius,window,dim,s,m,a,d0,d1,d2);
        float2 ds[3]={d0,d1,d2}, da=float2(0,0);
        for(int d=0;d<dim;d++) da+=ds[d]*delta_centers[s,d];
        sum+=cmul(da,float2(coeff[s,0],coeff[s,1]))
            +cmul(a,float2(delta_coeff[s,0],delta_coeff[s,1]));
    }
    output[m,0]=sum.x; output[m,1]=sum.y;
}
