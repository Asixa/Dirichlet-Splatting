// Float64 preprocessing for the range-response SVD. Rendering remains float32.
[AutoPyBindCUDA] [CUDAKernel]
void range_response(TensorView<double> output, double low, double high,
    int count, int bins, int samples, int nfft)
{
    uint index = cudaBlockIdx().x * 256 + cudaThreadIdx().x;
    if (index >= uint(count * bins)) return;
    uint bin = index / count, peak = index % count;
    double center = low + (high - low) * double(peak) / double(count - 1);
    double x = double(bin) - center;
    x -= double(nfft) * round(x / double(nfft));
    double pi = 3.1415926535897932384626433832795L;
    double a = pi * double(samples) / double(nfft), b = pi / double(nfft);
    double h = pi * double(samples - 1) / double(nfft);
    double ratio;
    if (abs(x) < 0.001L)
    {
        double c2 = (b*b - a*a) / 6.0L;
        double c4 = (3.0L*a*a*a*a + 7.0L*b*b*b*b - 10.0L*a*a*b*b) / 360.0L;
        ratio = 1.0L + c2*x*x + c4*x*x*x*x;
    }
    else
        ratio = sin(a*x) / (double(samples) * sin(b*x));
    output[bin, peak, 0] = ratio * cos(h*x);
    output[bin, peak, 1] = -ratio * sin(h*x);
}
