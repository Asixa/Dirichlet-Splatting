// Bounded reductions.

groupshared float energy_partial[8];
[AutoPyBindCUDA] [CUDAKernel]
void complex_energy(TensorView<float> value, TensorView<float> reference,
    TensorView<float> output, int count)
{
    int lane=cudaThreadIdx().x, i=cudaBlockIdx().x*256+lane;
    float square=0;
    if(i<count) {
        float2 z=float2(value[i,0]-reference[i,0],value[i,1]-reference[i,1]);
        square=dot(z,z);
    }
    float sum=WaveActiveSum(square);
    if(lane%32==0) energy_partial[lane/32]=sum;
    GroupMemoryBarrierWithGroupSync();
    if(lane<32) {
        sum=WaveActiveSum(lane<8 ? energy_partial[lane] : 0.0f);
        if(lane==0) output[cudaBlockIdx().x]=sum;
    }
}
