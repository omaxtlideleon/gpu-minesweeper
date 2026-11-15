# Results

## November 14, 2026

Ran the full board set five times on the RTX 4060. Every run matched all nine counts.

| Run | Billion boards/s | Seconds |
| ---: | ---: | ---: |
| 1 | 78.679 | 1.925 |
| 2 | 78.673 | 1.925 |
| 3 | 78.303 | 1.934 |
| 4 | 78.275 | 1.935 |
| 5 | 78.282 | 1.935 |

The median was **78.303 billion boards/s**, with a range of **78.275–78.679**. Each run covered all 151,473,214,816 boards. The verifier also passed all its board and neighbor checks. The output and CSV are in `results/`.

## Speed

A full run takes about **1.93 seconds**. The timer starts just before launching the GPU work and stops after copying the nine totals back to the CPU. Memory allocation and copying the 5,720-byte table happen before that. Finding each thread's starting board is part of the timed GPU work.

The build used CUDA 13.0.48, Visual Studio's x64 build environment, and `sm_89`. The main kernel used **40 registers with no spills**.

## Why the counts add up

There are `C(64,10) = 151,473,214,816` ways to put ten mines on 64 cells. The program goes through all of them once. That means there's no sampling error in the final counts.

The order is fixed, and nearby boards can look very similar. The result depends on finishing the whole run. A short chunk of the sequence isn't a random sample. The board count is fixed in the source, with a compile-time check to catch accidental changes.

The GPU builds and counts each board. After that, a separate formula checks the final totals.

For a safe cell with d neighbors and k mines next to it, the number of boards is:

```text
C(d, k) * C(63 - d, 10 - k)
```

Pick k mines from the neighbors, then put the remaining mines in the other cells. Add this up for 4 corners with 3 neighbors, 24 edge cells with 5 neighbors, and 36 inside cells with 8 neighbors.

| Mines next to a safe cell | Total appearances |
| ---: | ---: |
| 0 | 2,606,835,066,264 |
| 1 | 3,286,482,376,320 |
| 2 | 1,718,041,685,580 |
| 3 | 482,808,902,400 |
| 4 | 77,911,023,960 |
| 5 | 7,123,152,960 |
| 6 | 343,783,440 |
| 7 | 7,555,680 |
| 8 | 53,460 |

These add up to `54 * C(64,10)`, since every board has 54 safe cells. The printed percentages use all 64 cells as the denominator, so they add up to 84.375%.

## What was checked

The saved checks are in `results/verification.txt`:

- **6,451,712 boards** with ten mines and the right position in the sequence, checked separately on the CPU. This uses the next-board step and includes carries out of the lower 32 bits.
- All **12,582,912 thread intervals** joined up with no gaps or overlap. Board sequences were checked at sampled interval boundaries across the grid, plus other positions.
- **8,192 short runs** through the counting code matched a simple CPU count, including all nine buckets and the saved row counts.
- **3,377,280 cell neighbor counts** matched the CPU version, including every local mine pattern at every cell.
- All nine final totals matched the formula below on every full benchmark run.

## What makes it fast

The board order means the upper rows often stay unchanged for a while. A cell in the top half only depends on rows 3 through 7, counting rows from zero. If those rows haven't changed, its neighbor count hasn't changed either.

So the program counts the bottom half each time and saves the top half's counts. When those rows finally change, it multiplies the saved counts by the number of boards they applied to and adds them to the totals. Every board still gets generated and included once.

The next-board step uses 32-bit operations when the change fits in the lower half. If it carries into the upper half, it uses the full 64-bit step. A GPU shift instruction handles the case where the shift count is 32 without relying on an undefined C++ shift.

The loop is unrolled eight times, and the GPU gets small chunks of work. The counts for 7 and 8 are calculated directly. The main kernel uses **40 registers with no spills**.

See [README.md](README.md) for the build commands. `benchmark.ps1` runs the full program five times by default and checks the counts each time.
