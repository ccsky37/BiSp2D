#include "bisp2d.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static void checkCuda(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void warmGpuKernel(uint32_t *sink)
{
    uint64_t start = clock64();
    uint32_t value = blockIdx.x * blockDim.x + threadIdx.x + 1;
    while(clock64() - start < 200000000ULL)
    {
        value = value * 1664525U + 1013904223U;
    }
    if(threadIdx.x == 0)
    {
        atomicXor(sink, value);
    }
}

__global__ void buildBaSCWarpKernel(
    int rows,
    int cols,
    int paddedCols,
    const int8_t *__restrict__ dense,
    uint64_t *__restrict__ basc)
{
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int packedRow = blockIdx.y * blockDim.y + warp;
    int col = blockIdx.x * 32 + lane;
    int packedRows = (rows + 63) / 64;

    if(packedRow >= packedRows || col >= paddedCols)
    {
        return;
    }

    uint64_t value = 0;
    int rowStart = packedRow * 64;

    #pragma unroll
    for(int r = 0; r < 64; ++r)
    {
        int sourceRow = rowStart + r;
        if(sourceRow < rows && col < cols)
        {
            value = (value << 1) | static_cast<uint64_t>(dense[static_cast<size_t>(sourceRow) * cols + col]);
        }
    }

    basc[static_cast<size_t>(packedRow) * paddedCols + col] = value;
}

template<int G>
__global__ void buildBaSCGroupKernel(
    int rows,
    int cols,
    int paddedCols,
    const int8_t *__restrict__ dense,
    uint64_t *__restrict__ basc)
{
    int packedRow = blockIdx.y * blockDim.y + threadIdx.y;
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int colStart = group * G;
    int packedRows = (rows + 63) / 64;

    if(packedRow >= packedRows || colStart >= paddedCols)
    {
        return;
    }

    uint64_t values[G] = {0};
    int rowStart = packedRow * 64;

    #pragma unroll
    for(int r = 0; r < 64; ++r)
    {
        int sourceRow = rowStart + r;
        if(sourceRow < rows)
        {
            #pragma unroll
            for(int j = 0; j < G; ++j)
            {
                if(colStart + j < cols)
                {
                    values[j] = (values[j] << 1) |
                                static_cast<uint64_t>(dense[static_cast<size_t>(sourceRow) * cols + colStart + j]);
                }
            }
        }
    }

    #pragma unroll
    for(int j = 0; j < G; ++j)
    {
        if(colStart + j < paddedCols)
        {
            basc[static_cast<size_t>(packedRow) * paddedCols + colStart + j] = values[j];
        }
    }
}

template<int G>
static void launchGroupBuild(
    int rows,
    int cols,
    int paddedCols,
    const int8_t *dense,
    uint64_t *basc,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (paddedCols + block.x * G - 1) / (block.x * G),
        ((rows + 63) / 64 + block.y - 1) / block.y);
    buildBaSCGroupKernel<G><<<grid, block, 0, stream>>>(rows, cols, paddedCols, dense, basc);
}

static void launchBuildBaSC(
    int rows,
    int cols,
    int paddedCols,
    const int8_t *dense,
    uint64_t *basc,
    int g,
    cudaStream_t stream)
{
    if(g == 0)
    {
        dim3 block(32, 8);
        dim3 grid((paddedCols + 31) / 32, ((rows + 63) / 64 + block.y - 1) / block.y);
        buildBaSCWarpKernel<<<grid, block, 0, stream>>>(rows, cols, paddedCols, dense, basc);
        return;
    }

    switch(g)
    {
        case 1: launchGroupBuild<1>(rows, cols, paddedCols, dense, basc, stream); break;
        case 2: launchGroupBuild<2>(rows, cols, paddedCols, dense, basc, stream); break;
        case 4: launchGroupBuild<4>(rows, cols, paddedCols, dense, basc, stream); break;
        case 8: launchGroupBuild<8>(rows, cols, paddedCols, dense, basc, stream); break;
        case 16: launchGroupBuild<16>(rows, cols, paddedCols, dense, basc, stream); break;
        case 32: launchGroupBuild<32>(rows, cols, paddedCols, dense, basc, stream); break;
    }
}

__device__ __forceinline__ int countBits(uint64_t value)
{
    return __popcll(value);
}

__device__ __forceinline__ uint64_t wordAt(ulonglong4 value, int lane)
{
    if(lane == 0)
    {
        return value.x;
    }
    if(lane == 1)
    {
        return value.y;
    }
    if(lane == 2)
    {
        return value.z;
    }
    return value.w;
}

template<bool CheckK, bool CheckColumn>
__device__ __forceinline__ void loadTile(
    int kOffset,
    int packedRows,
    int groupsPerRow,
    uint tileRow,
    uint tileCol,
    const ulonglong4 *__restrict__ basc,
    ulonglong4 *left,
    ulonglong4 *right)
{
    const int blockSize = 32;
    const int groupsPerTile = 8;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for(int offset = tid; offset < blockSize * groupsPerTile; offset += blockDim.x * blockDim.y)
    {
        int row = offset / groupsPerTile;
        int group = offset - row * groupsPerTile;
        int packedRow = kOffset + row;
        int leftGroup = tileRow * groupsPerTile + group;
        int rightGroup = tileCol * groupsPerTile + group;
        bool validK = true;

        if constexpr (CheckK)
        {
            validK = packedRow < packedRows;
        }

        ulonglong4 leftValue = {0, 0, 0, 0};
        ulonglong4 rightValue = {0, 0, 0, 0};

        if(validK)
        {
            if constexpr (CheckColumn)
            {
                if(leftGroup < groupsPerRow)
                {
                    leftValue = basc[static_cast<size_t>(packedRow) * groupsPerRow + leftGroup];
                }
                if(rightGroup < groupsPerRow)
                {
                    rightValue = basc[static_cast<size_t>(packedRow) * groupsPerRow + rightGroup];
                }
            }
            else
            {
                leftValue = basc[static_cast<size_t>(packedRow) * groupsPerRow + leftGroup];
                rightValue = basc[static_cast<size_t>(packedRow) * groupsPerRow + rightGroup];
            }
        }

        left[offset] = leftValue;
        right[offset] = rightValue;
    }
}

template<bool SkipZero>
__device__ __forceinline__ void computeTile(
    uint threadRow,
    uint threadCol,
    const ulonglong4 *left,
    const ulonglong4 *right,
    uint *results)
{
    const int blockSize = 32;
    const int groupsPerTile = 8;

    #pragma unroll 2
    for(int k = 0; k < blockSize; ++k)
    {
        ulonglong4 leftGroup = left[k * groupsPerTile + threadRow];
        ulonglong4 rightGroup = right[k * groupsPerTile + (threadCol >> 1)];
        int lane = (threadCol & 1) << 1;
        uint64_t leftValues[4] = {leftGroup.x, leftGroup.y, leftGroup.z, leftGroup.w};
        uint64_t rightValues[2] = {wordAt(rightGroup, lane), wordAt(rightGroup, lane + 1)};

        if constexpr (SkipZero)
        {
            uint64_t leftMask = leftValues[0] | leftValues[1] | leftValues[2] | leftValues[3];
            uint64_t rightMask = rightValues[0] | rightValues[1];
            if(!leftMask || !rightMask)
            {
                continue;
            }
        }

        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            #pragma unroll
            for(int j = 0; j < 2; ++j)
            {
                results[i * 2 + j] += countBits(leftValues[i] & rightValues[j]);
            }
        }
    }
}

__device__ __forceinline__ void getTriangularTile(uint &tileRow, uint &tileCol)
{
    uint tileId = blockIdx.x;
    tileRow = static_cast<uint>((sqrtf(8.0f * tileId + 1.0f) - 1.0f) * 0.5f);
    uint rowStart = tileRow * (tileRow + 1) / 2;

    if(rowStart > tileId)
    {
        --tileRow;
        rowStart = tileRow * (tileRow + 1) / 2;
    }
    while(rowStart + tileRow + 1 <= tileId)
    {
        rowStart += tileRow + 1;
        ++tileRow;
    }
    tileCol = tileId - rowStart;
}

template<bool CheckColumn>
__device__ __forceinline__ void storeResult(
    uint32_t *output,
    int cols,
    uint row,
    uint col,
    uint value)
{
    if constexpr (CheckColumn)
    {
        if(row >= static_cast<uint>(cols) || col >= static_cast<uint>(cols))
        {
            return;
        }
    }

    output[static_cast<size_t>(row) * cols + col] = value;
    if(row != col)
    {
        output[static_cast<size_t>(col) * cols + row] = value;
    }
}

template<bool CheckColumn, bool SkipZero>
__global__ __launch_bounds__(128) void bascGemmKernel(
    int packedRows,
    int groupsPerRow,
    int cols,
    const ulonglong4 *__restrict__ basc,
    uint32_t *__restrict__ output)
{
    const int blockSize = 32;
    const int groupsPerTile = 8;
    uint tileRow;
    uint tileCol;
    getTriangularTile(tileRow, tileCol);

    uint threadRow = threadIdx.y;
    uint threadCol = threadIdx.x;
    uint rowBase = tileRow * blockSize + threadRow * 4;
    uint colBase = tileCol * blockSize + threadCol * 2;
    bool skipThread = tileRow == tileCol && rowBase + 4 <= colBase;

    __shared__ ulonglong4 left[blockSize * groupsPerTile];
    __shared__ ulonglong4 right[blockSize * groupsPerTile];
    uint results[8] = {0};
    int fullK = packedRows & ~(blockSize - 1);

    for(int k = 0; k < fullK; k += blockSize)
    {
        loadTile<false, CheckColumn>(k, packedRows, groupsPerRow, tileRow, tileCol, basc, left, right);
        __syncthreads();

        if(!skipThread)
        {
            computeTile<SkipZero>(threadRow, threadCol, left, right, results);
        }
        __syncthreads();
    }

    if(fullK < packedRows)
    {
        loadTile<true, CheckColumn>(fullK, packedRows, groupsPerRow, tileRow, tileCol, basc, left, right);
        __syncthreads();

        if(!skipThread)
        {
            computeTile<SkipZero>(threadRow, threadCol, left, right, results);
        }
        __syncthreads();
    }

    if(!skipThread)
    {
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            #pragma unroll
            for(int j = 0; j < 2; ++j)
            {
                storeResult<CheckColumn>(output, cols, rowBase + i, colBase + j, results[i * 2 + j]);
            }
        }
    }
}

static void launchBaSCGemm(
    int packedRows,
    int groupsPerRow,
    int cols,
    const uint64_t *basc,
    uint32_t *output,
    bool skipZero,
    cudaStream_t stream)
{
    int tileBlocks = (cols + 31) / 32;
    dim3 grid(tileBlocks * (tileBlocks + 1) / 2);
    dim3 block(16, 8);
    bool checkColumn = cols % 32 != 0;
    const ulonglong4 *vectorizedBaSC = reinterpret_cast<const ulonglong4 *>(basc);

    if(skipZero)
    {
        if(checkColumn)
        {
            bascGemmKernel<true, true><<<grid, block, 0, stream>>>(packedRows, groupsPerRow, cols, vectorizedBaSC, output);
        }
        else
        {
            bascGemmKernel<false, true><<<grid, block, 0, stream>>>(packedRows, groupsPerRow, cols, vectorizedBaSC, output);
        }
    }
    else
    {
        if(checkColumn)
        {
            bascGemmKernel<true, false><<<grid, block, 0, stream>>>(packedRows, groupsPerRow, cols, vectorizedBaSC, output);
        }
        else
        {
            bascGemmKernel<false, false><<<grid, block, 0, stream>>>(packedRows, groupsPerRow, cols, vectorizedBaSC, output);
        }
    }
}

struct IterationTiming
{
    float h2d;
    float build;
    float preparation;
    float gemm;
    float total;
};

static IterationTiming runOnce(
    const int8_t *input,
    int rows,
    int cols,
    int paddedCols,
    int packedRows,
    int groupsPerRow,
    int g,
    bool skipZero,
    int8_t *dense,
    uint64_t *basc,
    uint32_t *output,
    int streamsCount,
    cudaStream_t *streams,
    cudaStream_t mainStream,
    cudaStream_t phaseStream,
    cudaEvent_t start,
    cudaEvent_t allCopiesDone,
    cudaEvent_t gemmStart,
    cudaEvent_t gemmDone,
    cudaEvent_t stop,
    cudaEvent_t *copyStart,
    cudaEvent_t *copyDone,
    cudaEvent_t *buildStart,
    cudaEvent_t *buildDone)
{
    checkCuda(cudaEventRecord(start, mainStream));

    int packedBase = 0;
    for(int i = 0; i < streamsCount; ++i)
    {
        int packedCount = packedRows / streamsCount + (i < packedRows % streamsCount);
        int rowStart = packedBase * 64;
        int rowsThisStream = min(rows - rowStart, packedCount * 64);
        size_t denseOffset = static_cast<size_t>(rowStart) * cols;
        size_t denseBytes = static_cast<size_t>(rowsThisStream) * cols * sizeof(int8_t);
        size_t bascOffset = static_cast<size_t>(packedBase) * paddedCols;

        checkCuda(cudaStreamWaitEvent(streams[i], start));
        checkCuda(cudaEventRecord(copyStart[i], streams[i]));
        checkCuda(cudaMemcpyAsync(dense + denseOffset, input + denseOffset, denseBytes, cudaMemcpyHostToDevice, streams[i]));
        checkCuda(cudaEventRecord(copyDone[i], streams[i]));
        checkCuda(cudaEventRecord(buildStart[i], streams[i]));
        launchBuildBaSC(rowsThisStream, cols, paddedCols, dense + denseOffset, basc + bascOffset, g, streams[i]);
        checkCuda(cudaEventRecord(buildDone[i], streams[i]));
        packedBase += packedCount;
    }

    checkCuda(cudaStreamWaitEvent(phaseStream, start));
    for(int i = 0; i < streamsCount; ++i)
    {
        checkCuda(cudaStreamWaitEvent(phaseStream, copyDone[i]));
        checkCuda(cudaStreamWaitEvent(mainStream, buildDone[i]));
    }
    checkCuda(cudaEventRecord(allCopiesDone, phaseStream));
    checkCuda(cudaEventRecord(gemmStart, mainStream));
    launchBaSCGemm(packedRows, groupsPerRow, cols, basc, output, skipZero, mainStream);
    checkCuda(cudaEventRecord(gemmDone, mainStream));
    checkCuda(cudaEventRecord(stop, mainStream));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventSynchronize(allCopiesDone));
    checkCuda(cudaGetLastError());

    IterationTiming timing{};
    checkCuda(cudaEventElapsedTime(&timing.h2d, start, allCopiesDone));
    checkCuda(cudaEventElapsedTime(&timing.preparation, start, gemmStart));
    checkCuda(cudaEventElapsedTime(&timing.gemm, gemmStart, gemmDone));
    checkCuda(cudaEventElapsedTime(&timing.total, start, stop));

    for(int i = 0; i < streamsCount; ++i)
    {
        float elapsed = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed, buildStart[i], buildDone[i]));
        timing.build = max(timing.build, elapsed);
    }

    return timing;
}

static bool verifyResult(const int8_t *input, int rows, int cols, const uint32_t *output)
{
    for(int row = 0; row < cols; ++row)
    {
        for(int col = row + 1; col < cols; ++col)
        {
            if(output[static_cast<size_t>(row) * cols + col] != output[static_cast<size_t>(col) * cols + row])
            {
                return false;
            }
        }
    }

    for(int sample = 0; sample < 32; ++sample)
    {
        int left = (sample * 131 + 7) % cols;
        int right = (sample * 197 + 11) % cols;
        uint32_t expected = 0;
        for(int row = 0; row < rows; ++row)
        {
            expected += input[static_cast<size_t>(row) * cols + left] & input[static_cast<size_t>(row) * cols + right];
        }
        if(output[static_cast<size_t>(left) * cols + right] != expected)
        {
            return false;
        }
    }

    return true;
}

BiSp2DResult runBiSp2D(
    const int8_t *input,
    int rows,
    int cols,
    double sparsity,
    int g,
    int warmups,
    int runs)
{
    const int maxStreams = 4;
    int packedRows = (rows + 63) / 64;
    int paddedCols = (cols + 3) / 4 * 4;
    int groupsPerRow = paddedCols / 4;
    int streamsCount = min(maxStreams, packedRows);
    bool skipZero = sparsity > 99.0;

    int8_t *dense = nullptr;
    uint64_t *basc = nullptr;
    uint32_t *output = nullptr;
    checkCuda(cudaMalloc(&dense, static_cast<size_t>(rows) * cols * sizeof(int8_t)));
    checkCuda(cudaMalloc(&basc, static_cast<size_t>(packedRows) * paddedCols * sizeof(uint64_t)));
    checkCuda(cudaMalloc(&output, static_cast<size_t>(cols) * cols * sizeof(uint32_t)));

    cudaStream_t streams[maxStreams];
    cudaStream_t mainStream;
    cudaStream_t phaseStream;
    checkCuda(cudaStreamCreate(&mainStream));
    checkCuda(cudaStreamCreate(&phaseStream));

    cudaEvent_t start;
    cudaEvent_t allCopiesDone;
    cudaEvent_t gemmStart;
    cudaEvent_t gemmDone;
    cudaEvent_t stop;
    cudaEvent_t copyStart[maxStreams];
    cudaEvent_t copyDone[maxStreams];
    cudaEvent_t buildStart[maxStreams];
    cudaEvent_t buildDone[maxStreams];
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&allCopiesDone));
    checkCuda(cudaEventCreate(&gemmStart));
    checkCuda(cudaEventCreate(&gemmDone));
    checkCuda(cudaEventCreate(&stop));

    for(int i = 0; i < streamsCount; ++i)
    {
        checkCuda(cudaStreamCreate(&streams[i]));
        checkCuda(cudaEventCreate(&copyStart[i]));
        checkCuda(cudaEventCreate(&copyDone[i]));
        checkCuda(cudaEventCreate(&buildStart[i]));
        checkCuda(cudaEventCreate(&buildDone[i]));
    }

    warmGpuKernel<<<108, 256, 0, mainStream>>>(output);
    checkCuda(cudaStreamSynchronize(mainStream));
    checkCuda(cudaGetLastError());

    for(int i = 0; i < warmups; ++i)
    {
        runOnce(input, rows, cols, paddedCols, packedRows, groupsPerRow, g, skipZero,
                dense, basc, output, streamsCount, streams, mainStream, phaseStream,
                start, allCopiesDone, gemmStart, gemmDone, stop,
                copyStart, copyDone, buildStart, buildDone);
    }

    BiSp2DTiming mean{};
    for(int i = 0; i < runs; ++i)
    {
        IterationTiming value = runOnce(input, rows, cols, paddedCols, packedRows, groupsPerRow, g, skipZero,
                                        dense, basc, output, streamsCount, streams, mainStream, phaseStream,
                                        start, allCopiesDone, gemmStart, gemmDone, stop,
                                        copyStart, copyDone, buildStart, buildDone);
        mean.h2dMs += value.h2d;
        mean.buildMs += value.build;
        mean.preparationMs += value.preparation;
        mean.gemmMs += value.gemm;
        mean.totalMs += value.total;
    }

    mean.h2dMs /= runs;
    mean.buildMs /= runs;
    mean.preparationMs /= runs;
    mean.gemmMs /= runs;
    mean.totalMs /= runs;

    vector<uint32_t> hostOutput(static_cast<size_t>(cols) * cols);
    checkCuda(cudaMemcpy(hostOutput.data(), output, hostOutput.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bool valid = verifyResult(input, rows, cols, hostOutput.data());

    for(int i = 0; i < streamsCount; ++i)
    {
        checkCuda(cudaEventDestroy(copyStart[i]));
        checkCuda(cudaEventDestroy(copyDone[i]));
        checkCuda(cudaEventDestroy(buildStart[i]));
        checkCuda(cudaEventDestroy(buildDone[i]));
        checkCuda(cudaStreamDestroy(streams[i]));
    }
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(allCopiesDone));
    checkCuda(cudaEventDestroy(gemmStart));
    checkCuda(cudaEventDestroy(gemmDone));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaStreamDestroy(mainStream));
    checkCuda(cudaStreamDestroy(phaseStream));
    checkCuda(cudaFree(dense));
    checkCuda(cudaFree(basc));
    checkCuda(cudaFree(output));

    return {mean, valid};
}
