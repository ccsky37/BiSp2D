#include "bisp2d.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

static uint32_t nextRandom(uint32_t &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main(int argc, char **argv)
{
    int rows = argc > 1 ? atoi(argv[1]) : 20000;
    int cols = argc > 2 ? atoi(argv[2]) : 4096;
    double sparsity = argc > 3 ? atof(argv[3]) : 90.0;
    int g = argc > 4 ? atoi(argv[4]) : 0;
    int warmups = argc > 5 ? atoi(argv[5]) : 5;
    int runs = argc > 6 ? atoi(argv[6]) : 5;
    bool softwareCounter = argc > 7 && string(argv[7]) == "software";

    if(g != 0 && g != 1 && g != 2 && g != 4 && g != 8 && g != 16 && g != 32)
    {
        cerr << "g must be 0, 1, 2, 4, 8, 16, or 32\n";
        return 1;
    }

    size_t elements = static_cast<size_t>(rows) * cols;
    int8_t *input = nullptr;
    if(cudaMallocHost(&input, elements * sizeof(int8_t)) != cudaSuccess)
    {
        return 1;
    }

    double density = (100.0 - sparsity) / 100.0;
    uint32_t threshold = static_cast<uint32_t>(density * 4294967295.0);
    uint32_t state = 2026;
    for(size_t i = 0; i < elements; ++i)
    {
        input[i] = nextRandom(state) < threshold;
    }

    BiSp2DResult result = runBiSp2D(input, rows, cols, sparsity, g, warmups, runs, softwareCounter);

    cout << fixed << setprecision(3);
    cout << "Matrix: " << rows << " x " << cols << '\n';
    cout << "Sparsity: " << sparsity << "%\n";
    if(g == 0)
    {
        cout << "Build BaSC: warp\n";
    }
    else
    {
        cout << "Build BaSC: grouped, g=" << g << '\n';
    }
    cout << "BaSC-GEMM: 4x2 symmetric, "
         << (softwareCounter ? "software single-bit counter" : "native __popcll()")
         << ", zero skip "
         << (sparsity > 99.0 ? "enabled" : "disabled") << '\n';
    cout << "H2D: " << result.timing.h2dMs << " ms\n";
    cout << "Build BaSC: " << result.timing.buildMs << " ms\n";
    cout << "H2D + Build BaSC: " << result.timing.preparationMs << " ms\n";
    cout << "BaSC-GEMM: " << result.timing.gemmMs << " ms\n";
    cout << "Device-side Work: " << result.timing.buildMs + result.timing.gemmMs << " ms\n";
    cout << "GPU Total: " << result.timing.totalMs << " ms\n";
    cout << "Validation: " << (result.valid ? "PASS" : "FAIL") << '\n';

    cudaFreeHost(input);
    return result.valid ? 0 : 1;
}
