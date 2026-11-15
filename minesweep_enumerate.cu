#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <cuda_runtime.h>
// CUB adds up the totals from each block.
#include <cub/cub.cuh>

const long long TOTAL_SIMULATIONS = 151473214816LL; // Every C(64, 10) board.
const unsigned int NUM_MINES = 10;
const unsigned int WORKGROUP_SIZE = 256;
const unsigned int BLOCKS_PER_SM = 2048;
static_assert(NUM_MINES == 10 && TOTAL_SIMULATIONS == 151473214816LL,
              "This enumeration covers all C(64,10) boards.");

void checkCuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

__device__ __forceinline__ uint64_t majority64(uint64_t a, uint64_t b, uint64_t c) {
    // A bit is set if at least two inputs have it set. Each half uses
    // one LOP3 instruction, so the compiler does not have to figure it out.
    unsigned lo, hi;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(lo) : "r"((unsigned)a), "r"((unsigned)b), "r"((unsigned)c));
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(hi) : "r"((unsigned)(a>>32)), "r"((unsigned)(b>>32)), "r"((unsigned)(c>>32)));
    return (uint64_t(hi) << 32) | lo;
}
__device__ __forceinline__ void FA(uint64_t a, uint64_t b, uint64_t c, uint64_t & s, uint64_t & cout) {
    s = a ^ b ^ c;
    cout = majority64(a, b, c);
}

__device__ __forceinline__ void HA(uint64_t a, uint64_t b, uint64_t & s, uint64_t & cout) {
    s = a ^ b;
    cout = a & b;
}

__device__ __forceinline__ void neighborBits(uint64_t board, uint64_t & b0, uint64_t & b1, uint64_t & b2, uint64_t & b3) {
    // Add the left and right neighbors, then the center, to get a three-cell
    // row sum. Shift it up and down, then add the middle row's pair.
    // The center cell does not count itself.
    uint64_t w = (board & 0xFEFEFEFEFEFEFEFEULL) >> 1;
    uint64_t e = (board & 0x7F7F7F7F7F7F7F7FULL) << 1;
    uint64_t h0, h1;
    HA(w, e, h0, h1);
    uint64_t row0 = h0 ^ board;
    uint64_t row1 = h1 | (h0 & board);
    uint64_t carry0, tmp1, carry1, carry2;
    FA(row0 >> 8, row0 << 8, h0, b0, carry0);
    FA(row1 >> 8, row1 << 8, h1, tmp1, carry1);
    HA(tmp1, carry0, b1, carry2);
    HA(carry1, carry2, b2, b3);
}
__constant__ uint64_t chooseTable[65][11];

void initializeChooseTable() {
    uint64_t table[65][11] = {};
    for (int n = 0; n <= 64; ++n) {
        table[n][0] = 1;
        for (int k = 1; k <= 10; ++k)
            if (n > 0) table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(chooseTable, table, sizeof(table)));
}

uint64_t chooseHost(int n, int k) {
    if (k < 0 || k > n) return 0;
    uint64_t result = 1;
    for (int i = 1; i <= k; ++i) result = result * (n - k + i) / i;
    return result;
}

__device__ __forceinline__ uint64_t unrankBoard(uint64_t rank) {
    // Find the board at this position. Its rank is sum_j C(position_j, j).
    uint64_t board = 0;
    int upper = 63;
    for (int mines = 10; mines >= 1; --mines) {
        int lo = mines - 1, hi = upper;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (chooseTable[mid][mines] <= rank) lo = mid;
            else hi = mid - 1;
        }
        board |= 1ULL << lo;
        rank -= chooseTable[lo][mines];
        upper = lo - 1;
    }
    return board;
}

__device__ __forceinline__ uint64_t nextBoard(uint64_t board) {
    // Move the lowest group of set bits to get the next board.
    // Use a shift instead of dividing by the lowest set bit.
    uint64_t lowest = board & - board;
    uint64_t raised = board + lowest;
    unsigned shift = __ffsll(board) + 1;
    return raised | ((raised ^ board) >> shift);
}


__device__ __forceinline__ void countHalf(uint32_t empty, uint32_t b0, uint32_t b1,
                                         uint32_t b2, uint32_t b3, uint32_t* c) {
    c[1] += __popc(empty & b0);
    c[2] += __popc(empty & b1);
    c[3] += __popc(empty & b0 & b1);
    c[4] += __popc(empty & b2);
    c[5] += __popc(empty & b0 & b2);
    c[6] += __popc(empty & b1 & b2);
    uint32_t seven = empty & b0 & b1 & b2;
    {
        c[7] += __popc(seven);
        c[8] += __popc(empty & b3);
    }
}

__device__ __forceinline__ void countCached(uint64_t board, uint32_t* c,
                                            uint64_t& previous, uint32_t* cached, uint32_t& pending) {
    // The top half only sees mines in rows 3 through 7.
    // Save its counts until those rows change, then add the whole batch.
    uint64_t key = board >> 24;
    if (key != previous) {
        #pragma unroll
        for (int k = 1; k < 9; ++k) c[k] += cached[k] * pending;
        pending = 0;
        uint64_t b0, b1, b2, b3;
        neighborBits(key << 24, b0, b1, b2, b3);
        #pragma unroll
        for (int k = 0; k < 9; ++k) cached[k] = 0;
        countHalf(uint32_t(~(board >> 32)), uint32_t(b0 >> 32), uint32_t(b1 >> 32),
                  uint32_t(b2 >> 32), uint32_t(b3 >> 32), cached);
        previous = key;
    }
    uint64_t b0, b1, b2, b3;
    neighborBits(board, b0, b1, b2, b3);
    countHalf(uint32_t(~board), uint32_t(b0), uint32_t(b1), uint32_t(b2), uint32_t(b3), c);
    ++pending;
}


__device__ __forceinline__ uint64_t nextBoardFast(uint64_t board) {
    uint32_t low = uint32_t(board);
    uint32_t raised = low + (low & -low);
    if (raised != 0) {
        unsigned shift = __ffs(low) + 1;
        uint32_t tail;
        uint32_t changed = raised ^ low;
        asm("shr.u32 %0, %1, %2;" : "=r"(tail) : "r"(changed), "r"(shift));
        return (board & 0xffffffff00000000ULL) | (raised | tail);
    }
    return nextBoard(board);
}

__global__ void __launch_bounds__(256) fastSimulationKernel(unsigned long long * results) {
    uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t threads = uint64_t(gridDim.x) * blockDim.x;
    uint64_t pairs = uint64_t(TOTAL_SIMULATIONS) / 2;
    uint64_t base = pairs / threads, extra = pairs % threads;
    // Each thread gets its own chunk, with no skipped or repeated boards.
    uint64_t first = 2 * (idx * base + (idx < extra ? idx : extra));
    unsigned iterations = unsigned(base + (idx < extra));
    uint64_t board = unrankBoard(first);
    uint32_t c[9] = {0};
    uint32_t cached[9] = {0};
    uint64_t previous = ~uint64_t(0);
    uint32_t pending = 0;
    #pragma unroll 8
    for (unsigned i = 0; i < iterations; ++i) {
        uint64_t boardA = board;
        uint64_t boardB = nextBoardFast(boardA);
        board = nextBoardFast(boardB);
        countCached(boardA, c, previous, cached, pending);
        countCached(boardB, c, previous, cached, pending);
    }

    #pragma unroll
    for (int k = 1; k < 9; ++k) c[k] += cached[k] * pending;

    typedef cub::BlockReduce < uint64_t, WORKGROUP_SIZE > BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    #pragma unroll
    for (int i = 0; i < 9; i++) {
        uint64_t block_sum = BlockReduceT(temp_storage).Sum(c[i]);

        // Wait before reusing the shared scratch space.
        __syncthreads();

        // One thread adds the block's total to the final count.
        if (threadIdx.x == 0 && block_sum > 0) {
            atomicAdd( & results[i], (unsigned long long)block_sum);
        }

        // Keep the block together before the next bucket.
        __syncthreads();
    }
}

#ifndef MINESWEEP_TEST
int main() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice( & deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties( & props, deviceId));

    int numSMs = props.multiProcessorCount;
    unsigned int blocks = numSMs * BLOCKS_PER_SM;
    const uint64_t threads = uint64_t(blocks) * WORKGROUP_SIZE;
    const uint64_t pairs = uint64_t(TOTAL_SIMULATIONS) / 2;
    const uint64_t maxIterations = pairs / threads + (pairs % threads != 0);
    if (maxIterations > std::numeric_limits < uint32_t > ::max() / 108ULL) {
        std::cerr << "Board count exceeds the safe per-thread counter bound.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Using " << props.name << "\n";
    std::cout << "    Generating boards: " << TOTAL_SIMULATIONS << "\n\n";

    std::vector < unsigned long long > h_results(9, 0);
    unsigned long long * d_results = nullptr;
    CUDA_CHECK(cudaMalloc( & d_results, 9 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_results, 0, 9 * sizeof(unsigned long long)));

    initializeChooseTable();
    auto startTime = std::chrono::high_resolution_clock::now();

    fastSimulationKernel<<<blocks, WORKGROUP_SIZE>>>(d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 9 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    auto endTime = std::chrono::high_resolution_clock::now();

    // The GPU counts overlapping groups. Subtract the overlaps here.
    h_results[3] -= h_results[7];
    h_results[5] -= h_results[7];
    h_results[6] -= h_results[7];
    h_results[1] -= h_results[3] + h_results[5] + h_results[7];
    h_results[2] -= h_results[3] + h_results[6] + h_results[7];
    h_results[4] -= h_results[5] + h_results[6] + h_results[7];
    // Every board has 54 safe cells. Subtract counts 1 through 8 to get 0.
    // This saves another popcount for every board on the GPU.
    {
        unsigned long long sumRest = 0;
        for (int i = 1; i < 9; i++) sumRest += h_results[i];
        h_results[0] = 54ULL * (unsigned long long)TOTAL_SIMULATIONS - sumRest;
    }
    double timeTaken = std::chrono::duration < double > (endTime - startTime).count();

    // Check the totals with a separate formula. For a safe cell with
    // d neighbors, pick k mines there and 10-k in the other 63-d cells.
    for (int k = 0; k <= 8; ++k) {
        uint64_t expected = 4 * chooseHost(3, k) * chooseHost(60, 10 - k)
                           + 24 * chooseHost(5, k) * chooseHost(58, 10 - k)
                           + 36 * chooseHost(8, k) * chooseHost(55, 10 - k);
        if (h_results[k] != expected) {
            std::cerr << "histogram mismatch in bucket " << k << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "All nine histogram counts match the formula result.\n\n";

    unsigned long long totalBoards = TOTAL_SIMULATIONS;
    unsigned long long boardWidth = 8;
    unsigned long long boardHeight = 8;
    unsigned long long totalCellsSimulated = totalBoards * boardWidth * boardHeight;


    std::cout << "Results:" << std::endl;
    for (int i = 0; i < 9; i++) {
        double percentage = (static_cast < double > (h_results[i]) / totalCellsSimulated) * 100.0;
        std::cout << "    " << i << ": " << h_results[i] << " appearances ("
                  << std::fixed << std::setprecision(9) << percentage << "%)\n";
    }

    double boardSpeed = static_cast < double > (totalBoards) / timeTaken;
    std::cout << "\nExtra:" << std::endl;
    std::cout << "    Time taken: " << std::fixed << std::setprecision(3) << timeTaken << "s" << std::endl;
    std::cout << "    Speed: ~" << std::fixed << std::setprecision(0) << boardSpeed << " boards/s" << std::endl;
    std::cout << "    Speed (tiles): ~" << std::fixed << std::setprecision(1) << (boardSpeed * (boardWidth * boardHeight) / 1000000.0) << " Mtiles/s" << std::endl;


    CUDA_CHECK(cudaFree(d_results));
    return 0;
}
#endif
