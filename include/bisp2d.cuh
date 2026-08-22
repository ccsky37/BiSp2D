#pragma once

#include <cstdint>

struct BiSp2DTiming
{
    double h2dMs;
    double buildMs;
    double preparationMs;
    double gemmMs;
    double totalMs;
};

struct BiSp2DResult
{
    BiSp2DTiming timing;
    bool valid;
};

BiSp2DResult runBiSp2D(
    const int8_t *input,
    int rows,
    int cols,
    double sparsity,
    int g,
    int warmups,
    int runs);
