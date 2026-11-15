#define MINESWEEP_TEST
#include "minesweep_enumerate.cu"
#include <random>

constexpr unsigned SEQUENCE_LENGTH = 16;
__global__ void evaluateSequences(const uint64_t* ranks, uint64_t* boards, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t board = unrankBoard(ranks[i]);
    for (unsigned j = 0; j < SEQUENCE_LENGTH; ++j) {
        boards[i * SEQUENCE_LENGTH + j] = board;
        if (j + 1 < SEQUENCE_LENGTH) board = nextBoardFast(board);
    }
}


__global__ void evaluateCounts(const uint64_t* ranks, uint32_t* totals, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t c[9] = {}, cached[9] = {};
    uint64_t previous = ~uint64_t(0);
    uint32_t pending = 0;
    uint64_t board = unrankBoard(ranks[i]);
    for (unsigned j = 0; j < SEQUENCE_LENGTH; ++j) {
        countCached(board, c, previous, cached, pending);
        if (j + 1 < SEQUENCE_LENGTH) board = nextBoardFast(board);
    }
    #pragma unroll
    for (int k = 1; k < 9; ++k) c[k] += cached[k] * pending;
    c[3] -= c[7];
    c[5] -= c[7];
    c[6] -= c[7];
    c[1] -= c[3] + c[5] + c[7];
    c[2] -= c[3] + c[6] + c[7];
    c[4] -= c[5] + c[6] + c[7];
    c[0] = 54 * SEQUENCE_LENGTH;
    for (int k = 1; k < 9; ++k) c[0] -= c[k];
    for (int k = 0; k < 9; ++k) totals[i * 9 + k] = c[k];
}

void verifyEnumeration() {
    initializeChooseTable();
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    uint64_t threads = uint64_t(props.multiProcessorCount) * BLOCKS_PER_SM * WORKGROUP_SIZE;
    uint64_t pairs = uint64_t(TOTAL_SIMULATIONS) / 2;
    uint64_t base = pairs / threads, extra = pairs % threads;
    std::vector<uint64_t> ranks{0, 1, uint64_t(TOTAL_SIMULATIONS) - SEQUENCE_LENGTH};
    uint64_t previousEnd = 0;
    for (uint64_t t = 0; t < threads; ++t) {
        uint64_t first = 2 * (t * base + std::min(t, extra));
        uint64_t count = 2 * (base + (t < extra));
        if (first != previousEnd || count < SEQUENCE_LENGTH) std::exit(EXIT_FAILURE);
        previousEnd = first + count;
        // Check all interval bounds and sample board sequences across the grid.
        if (t % (BLOCKS_PER_SM / 32) == 0 || t + 1 == threads) {
            ranks.push_back(first);
            ranks.push_back(previousEnd - SEQUENCE_LENGTH);
        }
    }
    if (previousEnd != uint64_t(TOTAL_SIMULATIONS)) std::exit(EXIT_FAILURE);

    // Include carries out of the lower half, with different numbers of low bits.
    for (int lowMines = 0; lowMines <= 10; ++lowMines) {
        uint64_t low = lowMines ? ((1ULL << lowMines) - 1) << (32 - lowMines) : 0;
        uint64_t board = low | (((1ULL << (10 - lowMines)) - 1) << 32);
        uint64_t rank = 0;
        int mine = 0;
        for (int bit = 0; bit < 64; ++bit)
            if ((board >> bit) & 1) rank += chooseHost(bit, ++mine);
        if (rank + SEQUENCE_LENGTH <= uint64_t(TOTAL_SIMULATIONS)) ranks.push_back(rank);
    }
    std::mt19937_64 random(20261011);
    for (int i = 0; i < 10000; ++i)
        ranks.push_back(random() % (uint64_t(TOTAL_SIMULATIONS) - SEQUENCE_LENGTH + 1));
    uint64_t* dRanks = nullptr, *dBoards = nullptr;
    CUDA_CHECK(cudaMalloc(&dRanks, ranks.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dBoards, ranks.size() * SEQUENCE_LENGTH * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dRanks, ranks.data(), ranks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    evaluateSequences<<<unsigned((ranks.size()+255)/256), 256>>>(dRanks, dBoards, ranks.size());
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint64_t> boards(ranks.size() * SEQUENCE_LENGTH);
    CUDA_CHECK(cudaMemcpy(boards.data(), dBoards, boards.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t cpuChoose[64][11];
    for (int bit = 0; bit < 64; ++bit)
        for (int k = 0; k <= 10; ++k) cpuChoose[bit][k] = chooseHost(bit, k);
    for (size_t i = 0; i < ranks.size(); ++i) {
        for (unsigned j = 0; j < SEQUENCE_LENGTH; ++j) {
            uint64_t board = boards[i * SEQUENCE_LENGTH + j], actualRank = 0;
            unsigned mines = 0;
            for (int bit = 0; bit < 64; ++bit) {
                if ((board >> bit) & 1) {
                    if (++mines > 10) std::exit(EXIT_FAILURE);
                    actualRank += cpuChoose[bit][mines];
                }
            }
            if (mines != 10 || actualRank != ranks[i] + j) {
                std::cerr << "Enumeration mismatch at rank " << ranks[i]+j << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // Check the new counting path on short runs spread across the whole space.
    size_t checked = std::min(size_t(8192), ranks.size());
    std::vector<uint64_t> countRanks(checked);
    std::vector<size_t> indices(checked);
    for (size_t i = 0; i < checked; ++i) {
        indices[i] = i * (ranks.size() - 1) / (checked - 1);
        countRanks[i] = ranks[indices[i]];
    }
    uint64_t* dCountRanks = nullptr;
    uint32_t* dCounts = nullptr;
    CUDA_CHECK(cudaMalloc(&dCountRanks, checked * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dCounts, checked * 9 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dCountRanks, countRanks.data(), checked * sizeof(uint64_t), cudaMemcpyHostToDevice));
    evaluateCounts<<<unsigned((checked + 255) / 256), 256>>>(dCountRanks, dCounts, checked);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> counts(checked * 9);
    CUDA_CHECK(cudaMemcpy(counts.data(), dCounts, counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < checked; ++i) {
        uint32_t expected[9] = {};
        for (unsigned j = 0; j < SEQUENCE_LENGTH; ++j) {
            uint64_t board = boards[indices[i] * SEQUENCE_LENGTH + j];
            for (int cell = 0; cell < 64; ++cell) {
                if ((board >> cell) & 1) continue;
                int neighbors = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy) continue;
                        int x = cell % 8 + dx, y = cell / 8 + dy;
                        if (x >= 0 && x < 8 && y >= 0 && y < 8)
                            neighbors += (board >> (y * 8 + x)) & 1;
                    }
                }
                ++expected[neighbors];
            }
        }
        for (int k = 0; k < 9; ++k) {
            if (counts[i * 9 + k] != expected[k]) {
                std::cerr << "Counting mismatch at rank " << countRanks[i] << ", bucket " << k << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    }
    CUDA_CHECK(cudaFree(dCountRanks));
    CUDA_CHECK(cudaFree(dCounts));
    std::cout << "PASS: " << checked << " short runs match the CPU counts with row reuse enabled.\n";
    CUDA_CHECK(cudaFree(dRanks));
    CUDA_CHECK(cudaFree(dBoards));
    std::cout << "PASS: " << boards.size() << " GPU enumerated boards have ten mines and the expected independent CPU rank.\n"
              << "PASS: all " << threads << " thread intervals are contiguous, disjoint, and cover the full board space.\n";
}

__global__ void evaluateBoards(const uint64_t* boards, uint64_t* planes, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        neighborBits(boards[i], planes[4 * i], planes[4 * i + 1], planes[4 * i + 2], planes[4 * i + 3]);
    }
}

int main() {
    verifyEnumeration();
    std::vector<uint64_t> boards{0, ~uint64_t(0)};
    // Try every pattern of mines around each cell, including the edges,
    // with and without a mine in the center.
    for (int cell = 0; cell < 64; ++cell) {
        for (unsigned pattern = 0; pattern < 256; ++pattern) {
            uint64_t board = 0;
            unsigned bit = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    int x = cell % 8 + dx, y = cell / 8 + dy;
                    if (x >= 0 && x < 8 && y >= 0 && y < 8 && (pattern & (1u << bit)))
                        board |= uint64_t(1) << (y * 8 + x);
                    ++bit;
                }
            }
            boards.push_back(board);
            boards.push_back(board | (uint64_t(1) << cell));
        }
    }
    std::mt19937_64 random(20261007);
    for (int i = 0; i < 10000; ++i) {
        boards.push_back(random());
        uint64_t sparse = 0;
        for (int j = 0; j < 10; ++j) sparse |= uint64_t(1) << (random() & 63);
        boards.push_back(sparse);
    }
    uint64_t* input = nullptr, *output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, boards.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&output, boards.size() * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(input, boards.data(), boards.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    evaluateBoards<<<unsigned((boards.size()+255)/256), 256>>>(input, output, boards.size());
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint64_t> planes(boards.size() * 4);
    CUDA_CHECK(cudaMemcpy(planes.data(), output, planes.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < boards.size(); ++i) {
        for (int cell = 0; cell < 64; ++cell) {
            unsigned expected = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    int x = cell % 8 + dx, y = cell / 8 + dy;
                    if (x >= 0 && x < 8 && y >= 0 && y < 8)
                        expected += unsigned((boards[i] >> (y * 8 + x)) & 1);
                }
            }
            unsigned actual = 0;
            for (int b = 0; b < 4; ++b) actual |= unsigned((planes[i * 4 + b] >> cell) & 1) << b;
            if (actual != expected) {
                std::cerr << "Mismatch: board " << i << ", cell " << cell << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    std::cout << "PASS: " << boards.size() << " GPU boards / " << boards.size()*64
              << " cell counts match the scalar CPU reference.\n";
}
