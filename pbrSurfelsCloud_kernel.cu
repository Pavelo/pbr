#ifndef _PBRSURFELSCLOUD_KERNEL_H_
#define _PBRSURFELSCLOUD_KERNEL_H_

#define BLOCK_SIZE 4

///////////////////////////////////////////////////////////////////////////////
//! Kernel to copy a poly model from device to host
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void faceArea_kernel(int4* face_v_id, float3* vertex, float* face_area, float4* service)
{
	// thread id
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	// length of edges with formula of
	// euclidean distance = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )
	// ---
	// (x1-x2), (y1-y2), (z1-z2)
	// for each pair of face vertexes
	float3 a_comp = {
		vertex[ face_v_id[tid].x-1 ].x - vertex[ face_v_id[tid].y-1 ].x ,
		vertex[ face_v_id[tid].x-1 ].y - vertex[ face_v_id[tid].y-1 ].y ,
		vertex[ face_v_id[tid].x-1 ].z - vertex[ face_v_id[tid].y-1 ].z };
	float3 b_comp = {
		vertex[ face_v_id[tid].z-1 ].x - vertex[ face_v_id[tid].y-1 ].x ,
		vertex[ face_v_id[tid].z-1 ].y - vertex[ face_v_id[tid].y-1 ].y ,
		vertex[ face_v_id[tid].z-1 ].z - vertex[ face_v_id[tid].y-1 ].z };
	float3 c_comp = {
		vertex[ face_v_id[tid].z-1 ].x - vertex[ face_v_id[tid].w-1 ].x ,
		vertex[ face_v_id[tid].z-1 ].y - vertex[ face_v_id[tid].w-1 ].y ,
		vertex[ face_v_id[tid].z-1 ].z - vertex[ face_v_id[tid].w-1 ].z };
	float3 d_comp = {
		vertex[ face_v_id[tid].x-1 ].x - vertex[ face_v_id[tid].w-1 ].x ,
		vertex[ face_v_id[tid].x-1 ].y - vertex[ face_v_id[tid].w-1 ].y ,
		vertex[ face_v_id[tid].x-1 ].z - vertex[ face_v_id[tid].w-1 ].z };

	// complete formula
	// TODO: trasformare in variabile accessibile solo dal singolo thread o in shared mem
//	__shared__ float4 edge_length[ BLOCK_SIZE ];
//	edge_length[tid].x = sqrtf( a_comp.x*a_comp.x + a_comp.y*a_comp.y + a_comp.z*a_comp.z );
//	edge_length[tid].y = sqrtf( b_comp.x*b_comp.x + b_comp.y*b_comp.y + b_comp.z*b_comp.z );
//	edge_length[tid].z = sqrtf( c_comp.x*c_comp.x + c_comp.y*c_comp.y + c_comp.z*c_comp.z );
//	edge_length[tid].w = sqrtf( d_comp.x*d_comp.x + d_comp.y*d_comp.y + d_comp.z*d_comp.z );
	service[tid].x = sqrtf( a_comp.x*a_comp.x + a_comp.y*a_comp.y + a_comp.z*a_comp.z );
	service[tid].y = sqrtf( b_comp.x*b_comp.x + b_comp.y*b_comp.y + b_comp.z*b_comp.z );
	service[tid].z = sqrtf( c_comp.x*c_comp.x + c_comp.y*c_comp.y + c_comp.z*c_comp.z );
	service[tid].w = sqrtf( d_comp.x*d_comp.x + d_comp.y*d_comp.y + d_comp.z*d_comp.z );
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void faceArea(int n_faces, int4* face_v_id, float3* vertex, float* face_area, float4* service)
{
    // execute the kernel
    dim3 grid( 1, 1, 1 );
    dim3 block( n_faces, 1, 1 );
    faceArea_kernel<<< grid, block >>>(face_v_id, vertex, face_area, service);
}

#endif // #ifndef _PBRSURFELSCLOUD_KERNEL_H_
