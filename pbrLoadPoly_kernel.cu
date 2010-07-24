#ifndef _PBRSURFELSCLOUD_KERNEL_H_
#define _PBRSURFELSCLOUD_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Kernel to copy a poly model from device to host
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel()
{
	
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(int* input_mesh, int* output_mesh)
{
    // execute the kernel
    dim3 grid(1, 1, 1);
    dim3 block(10, 1, 1);
    kernel<<< grid, block>>>();
}

#endif // #ifndef _PBRSURFELSCLOUD_KERNEL_H_
