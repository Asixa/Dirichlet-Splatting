// Candidate kernels of the spatial search: range gating, separated peaks
// and the batched quasi-Newton step of the certificate refinement.

#include "common.slang"

// Spherical range gating; each candidate checks its own intervals.
[AutoPyBindCUDA] [CUDAKernel]
void candidate_range_gate(TensorView<float> points, TensorView<float> positions,
    TensorView<int> intervals, TensorView<int> output,
    int count, int apertures, int bins, float spacing, int required)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=count) return;
    int matches=0;
    for(int a=0;a<apertures;a++) {
        float3 delta=float3(points[i,0]-positions[a,0],points[i,1]-positions[a,1],points[i,2]-positions[a,2]);
        float x=length(delta)/spacing, lo=floor(x);
        int k=int(lo);
        // Match round-to-even used by the tensor reference at exact half bins.
        if(x-lo>0.5f || (x-lo==0.5f && (k%2)!=0)) k++;
        if(k>=0 && k<bins && intervals[a,k]!=0) matches++;
    }
    output[i]=matches>=required ? 1 : 0;
}

// Greedy separated peaks in one block.
groupshared float peak_values[256];
groupshared int peak_indices[256];
[AutoPyBindCUDA] [CUDAKernel]
void candidate_peaks(TensorView<float> scores, TensorView<float> centers,
    TensorView<int> output, int count, int peaks, int dimension, float separation)
{
    int lane=cudaThreadIdx().x;
    for(int p=0;p<peaks;p++) {
        float best=-1.0f/0.0f; int index=-1;
        for(int i=lane;i<count;i+=256) {
            float value=scores[i];
            if(isfinite(value) && (value>best || (value==best && (index<0 || i<index)))) {
                best=value; index=i;
            }
        }
        peak_values[lane]=best; peak_indices[lane]=index;
        GroupMemoryBarrierWithGroupSync();
        for(int stride=128;stride>0;stride/=2) {
            if(lane<stride) {
                float value=peak_values[lane+stride]; int other=peak_indices[lane+stride];
                if(other>=0 && (value>peak_values[lane] ||
                    (value==peak_values[lane] && (peak_indices[lane]<0 || other<peak_indices[lane])))) {
                    peak_values[lane]=value; peak_indices[lane]=other;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        index=peak_indices[0];
        if(lane==0) output[p]=index;
        if(index<0) return;
        for(int i=lane;i<count;i+=256) {
            float d2=0;
            for(int j=0;j<dimension;j++) { float d=centers[i,j]-centers[index,j]; d2+=d*d; }
            if(i==index || d2<separation*separation) scores[i]=-1.0f/0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// One thread per independent peak: fuse the small quasi-Newton direction,
// tangent projection and trust-radius scaling without host scalar reads.
[AutoPyBindCUDA] [CUDAKernel]
void candidate_direction(TensorView<float> inverse, TensorView<float> gradient,
    TensorView<float> normals, TensorView<float> mask, TensorView<float> output,
    int count, int width, float radius)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=count) return;
    float d[6]; float descent=0;
    for(int j=0;j<width;j++) {
        d[j]=0;
        for(int k=0;k<width;k++) d[j]-=inverse[i,j,k]*gradient[i,k];
        descent+=d[j]*gradient[i,j];
    }
    for(int j=0;j<width;j++) d[j]=(descent>=0 ? -gradient[i,j] : d[j])*mask[j];
    if(width==6) {
        float projection=0;
        for(int j=0;j<3;j++) projection+=d[j+3]*normals[i,j];
        for(int j=0;j<3;j++) d[j+3]-=projection*normals[i,j];
    }
    float length2=0;
    for(int j=0;j<width;j++) length2+=d[j]*d[j];
    float scale=min(1.0f,radius/max(sqrt(length2),1e-30f));
    for(int j=0;j<width;j++) output[i,j]=d[j]*scale;
}

[AutoPyBindCUDA] [CUDAKernel]
void candidate_trial(TensorView<float> centers, TensorView<float> normals,
    TensorView<float> direction, TensorView<float> controls,
    TensorView<float> out_centers, TensorView<float> out_normals,
    int count, int width, float fraction)
{
    int i=cudaBlockIdx().x*cudaBlockDim().x+cudaThreadIdx().x;
    if(i>=count) return;
    float n[3]; float length2=0;
    for(int j=0;j<3;j++) {
        float c=centers[i,j]+fraction*controls[0,j]*direction[i,j];
        out_centers[i,j]=controls[1,j]>0 ? clamp(c,controls[2,j],controls[3,j]) : centers[i,j];
        n[j]=normals[i,j]+(width==6 ? fraction*controls[0,j+3]*direction[i,j+3] : 0);
        length2+=n[j]*n[j];
    }
    for(int j=0;j<3;j++) out_normals[i,j]=width==6 ? n[j]/sqrt(length2) : normals[i,j];
}
