# GPU Minesweeper

I've been working on this Minesweeper project for about 4 years. It started in HTML and JavaScript, then went through multithreading, WebGPU, and eventually CUDA C++. This is the version that goes through every possible 8x8 board with exactly 10 mines.

There are **151,473,214,816** of those boards (`64 choose 10`). On my RTX 4060, this version went through all of them in about **1.93 seconds**, or **78.303 billion boards/s** at the median of five runs.

For each board, it counts how many safe cells have 0, 1, 2, ... 8 mines next to them. It adds those counts together on the GPU and checks the final answer against a separate math formula.

## Running it and/or building it

You'll need an NVIDIA GPU, the CUDA Toolkit, and a C++ compiler that works with CUDA. I built it on Windows with CUDA 13.0 and the Visual Studio 2022 C++ build tools. CUB comes with CUDA.

Open a terminal application with the CUDA Toolkit on PATH, go to this folder, and run:

```powershell
.\build.ps1
.\minesweep_enumerate.exe
```

The build script defaults to `sm_89`, which is what I used for my RTX 4060. If your GPU needs a different target, pass it with `-Arch`:

```powershell
.\build.ps1 -Arch sm_86
```

To build and run the extra checks:

```powershell
.\build.ps1 -Source verify_enumerate.cu -Output verify_enumerate.exe
.\verify_enumerate.exe
```

To run the full program five times and get a median speed:

```powershell
.\benchmark.ps1
```

You can also build directly with nvcc. These commands use the same GPU target, so change it if needed:

```text
nvcc -std=c++17 -O3 -arch=sm_89 minesweep_enumerate.cu -o minesweep_enumerate
nvcc -std=c++17 -O3 -arch=sm_89 verify_enumerate.cu -o verify_enumerate
```

## How it works

A board fits in one 64-bit integer: a 1 means a mine, and a 0 means a safe cell. Each GPU thread gets its own chunk of the possible boards. It finds the first board in that chunk, then uses a few bit operations to get the next integer with exactly ten set bits.

The neighbor counts use bit shifts and bitwise addition. The bottom half gets counted for each board. The top half often stays the same, so its counts are saved and added in batches until one of the rows it depends on changes. Each loop works on two boards, and the threads add their totals together at the end.

The boards come in a fixed order. Every board shows up once, so the final counts cover the whole set without sampling error.

## About the speed

The speed includes generating the boards and counting their neighbors on the GPU. Only the nine final counts get copied back to the CPU (it doesn't save 151 billion boards to RAM or disk)

The timer covers the GPU launch, waiting for it to finish, and copying those nine counts back. Allocating memory and setting up the small lookup table happen before the timer starts. Printing and the final math check happen afterward.

Your speed will depend on your GPU and what else is running. The saved runs and more details are in [RESULTS.md](RESULTS.md).

## Files

- `minesweep_enumerate.cu`: the main program
- `verify_enumerate.cu`: checks the board sequence and neighbor counts
- `build.ps1`: builds either program on Windows
- `benchmark.ps1`: runs the main program a few times and prints the median
- `RESULTS.md`: the numbers and how they were checked
- `results/`: saved output from the runs
