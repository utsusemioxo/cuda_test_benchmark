# Memory architecture and data locality

## Importance of memory access efficiency

- Global memory is typically implemented with off-chip DRAM, tends to have long access latency (hundreds of clock cycles) and finite access bandwidth. While having many threads available for execution can theoretically tolerate long memory access latencies, one can easily run into a situation in which traffic congestion in the global memory access paths prevents all but a very few threads from making progress, thus rendering some of the cores in the streaming multiprocessors (SMs) idle. To circumvent such congestion, GPUs provide a number of additional on-chip memory resources for accessing data that can remove the majority of traffic to and from the global memory.
- *compute to global memory access ratio* or *arithmetic intensity* or *computational intensity*: The ratio of floating-point operations (FLOP) to bytes (B) accessed from global memory.
- The compute to global memory access ratio has major implications for the performance of a CUDA kernel.
	- A100 GPU's peak global memory bandwidth is 1555GB/second
	- matrix multiplication kernel performs 0.25 OP/B
	- the global memory bandwidth limits the throughput of single-precision FLOPs that can be performed by the kernel to 389 giga FLOPs per second(GFLOPS)
	- however 389 GFLOPS is only 2% of the peak single-precision operation throughput of the A100 GPU, which is 19500 GFLOPS.
	- A100 also has special purpose units called *tensor cores* that are useful for accelerating matrix multiplication operations. If one considers the A100's tensor-core peak single-precision floating-point throughput of 156000 GFLOPS, 389 GFLOPS is only 0.25% of the peak.
- The programs whose execution speed is limited by memory bandwidth is called *memory-bound* programs.
	- $389 \, \text{GFLOPS} = 1555 \, \text{GB/S} \times 0.25 \, \text{OP/B}$ , we need to increase the compute to global memory access ratio of the kernel.
	- We can increase the compute to global memory access ratio by reducing the number of global memory access it performs.
	- To fully utilize the 19500 GFLOPS that the A100 GPU provides, a ratio of at least $(19500 \, \text{GOP/second}) / (1555\, \text{GB/second})=12.5\,\text{OP/B}$ is needed. This ratio means that for every 4-byte floating point value accessed, there must be about 50 floating-point operations performed.
	- The extent to which such a ratio can be achieved depends on the intrinsic data reuse in the computation at hand.

## The Roofline Model
![[roofline_model.png]]
## CUDA memory types
- A CUDA device contains types of memory that can help programmers to improve the compute to global memory access ratio.
![[cuda_memory_figure_5_2.png]]
1. Off-Chip Memory
	1. Constant Memory: Short-latency, high-bandwidth read-only access by device and host.
	2. Global Memory: Long-latency, low-bandwidth read and written by host and device.
	3. Local Memory: Placed in global memory, has similar access latency, but not shared across threads. Each thread has its own section of global memory that it uses as its own private local memory where it places data that is private to the thread but cannot be allocated in registers.
2. On-Chip Memory (Variables in Registers and shared memory can be accessed at very high speed in a highly parallel manner.)
	1. Registers: allocated to individual threads, each thread can access only its own registers. A kernel function typically uses registers to hold frequently accessed variables that are private to each thread.
	2. Shared Memory: allocated to thread blocks. All threads in a block can access shared memory variables declared for the block. Shared memory is an efficient means by which threads can cooperate by sharing their input data and intermediate results.
## CPU vs. GPU Register Architecture
1. Zero overhead scheduling
	- When CPUs context switch between different threads, they save the registers of the outgoing thread to memory and restore the registers of the incoming thread from memory.
	- In contrast, GPUs achieve zero-overhead scheduling by keeping the registers of all threads that are scheduled on the processing block in the processing block's register file. This way, switching between warps of threads is instantaneous because the registers of the incoming threads are already in the register file. Consequently, GPU register files need to be substantially larger than CPU register files.
2. Dynamic resource partitioning
	- GPU also support dynamic resource partitioning where an SM may provision few registers per thread and execute a large number of threads,  or it may provision more registers per thread and execute fewer threads. For this reason, GPU register files need to be designed to support such dynamic partitioning of registers. In contrast, the CPU register architecture dedicates a fixed set of registers per thread regardless of the thread's actual demand for registers.


## Memory versus registers
- The global memory is off the processor chip and is implemented with DRAM technology, which implies long access latencies and relatively low access bandwidth.
- The registers correspond to the "Register File" of the von Neumann model. 
	- The Register File is on the processor chip, which implies very short access latency and drastically higher access bandwidth when compared to the global memory. (at least two orders of magnitude higher.) 
	- Furthermore, whenever a variable is stored in a register, its access no longer consume off-chip global memory bandwidth, this will be reflected as an increased computed to global memory access ratio.
	- Each access to registers involves fewer instructions than an access to global memory. Arithmetic instructions in most modern processors have "built-in" register operands.
	- In modern computers the energy that is consumed for accessing a value from the register file is at least an order of magnitude lower than for accessing value from global memory. Accessing a value from registers has a tremendous advantage in energy efficiency over accessing the value from the global memory.
	- The number of registers that are available to each thread is limited in today's GPUs. The occupancy achieved for an application can be reduced if the register usage in full-occupancy scenarios exceeds the limit. Therefore we also need to avoid oversubscribing to this limited resource whenever possible.
![[von Neumann model.png]]
![[gpu_von_neumann_model.png]]
