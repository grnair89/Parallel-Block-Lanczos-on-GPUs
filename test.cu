#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda.h>
#include<pthread.h>
#include "cuPrintf.cu"
using namespace std;
#define N 32
#define BLOCK_SIZE 3584
#define NO_BLOCK 16
//__constant__ uint Yblock[4*BLOCK_SIZE];

char name[]="rsa120";
uint indices_no;
uint densewords;
uint dims[3];
//checked
void read_sizes()
{
	FILE* mat;
	char buf[256];
	sprintf(buf, "%s.dat.mat", name);
	mat=fopen(buf,"rb");
	for(int i=0;i<3;i++)
		fread(&dims[i],4,1,mat);
	//no_of_cols|dense_row_no|no_of_rows
	for(int i=0;i<3;i++)
		cout<<dims[i]<<endl;	
}
//checked
uint tell_size()
{
	uint n;
	FILE* mat;
	char buf[256];
	sprintf(buf, "%s_indices.dat", name);
	mat=fopen(buf,"rb");
	fseeko(mat,0,SEEK_END);
	n=ftello(mat);
	fclose(mat);
	return n;		
}
void read_matrix(uint *B,uint n) 
{
	
	FILE *mat;
	char buf[256];
	sprintf(buf, "%s_indices.dat", name);
	mat=fopen(buf,"rb");
	//indices_no=n/4;
	fread(B,sizeof(uint),(size_t)n,mat);
	cout<<"reached here\n";
	fclose(mat);
		
}
//checked
void read_matrix_offsets(uint *B_offsets,uint n)
{
	FILE *mat_offsets;
	char buf[256];
	sprintf(buf, "%s_strip.dat", name);
	mat_offsets=fopen(buf,"rb");
	uint x;
	for(int i=0;i<n;i++)
	{
		fread(&x,sizeof(uint),(size_t)1,mat_offsets);			
		B_offsets[i]=x;
	}
	indices_no=B_offsets[n-1];
	
}		

//checked
void read_matrix3(uint *B,uint n) {
	FILE *mat;
	char buf[256];
	sprintf(buf, "%s_dense.dat", name);
	mat=fopen(buf,"rb");
	fread(B,sizeof(uint),(size_t)n,mat);
	fclose(mat);	
}


__global__ void print(uint** B)
{
	for(uint i=0;i<52;i++)
		cuPrintf("%u ",B[i][0]);
	cuPrintf("\n");
}

__global__ void check_bcopy(uint* bcopy)
{
		for(int i=0;i<100;i++)	
		cuPrintf("%u ",bcopy[i]);
		cuPrintf("\n");
	
}

//__global__ void check_dense(uint* dense)
//{
//	for(uint i=0;i<100;i++)
//		cuPrintf("%u ",Yblock[i]);
//	cuPrintf("\n");
//}

//checked
void rand_gen(uint *Y,uint n)
{
	srand(time(NULL));
	cout<<"in rangen\n";
	for(uint i=0;i<n/3;i++)
	{
	    Y[i]=rand();
	    Y[i]+=100*i;
	}
	srand(time(NULL));
	for(uint i=n/3;i<2*n/3;i++)
	{
	    Y[i]=rand();
	    Y[i]+=177*i;
	}
	srand(time(NULL));
	for(uint i=2*n/3;i<n;i++)
	{
	    Y[i]=rand();
	    Y[i]+=39*i;
	}
}
//checked
void mul_NNNN(uint* a,uint *b,uint *c)
{
	uint x,col;
	memset(c,0,32*4);
	for(int i=0;i<32;i++)
	{
		col=0;
		x=a[i];
		while(x)
		{
			if(x&1)
				c[col]^=b[col];
			x>>=1;
			col++;
		}
	}
}
//checked
void add_NNNN(uint* a,uint*b,uint*c)
{
	if(c!=NULL && a!=NULL & b!=NULL)
	{
		for(int i=0;i<N;i++)
		{
			c[i]=a[i]^b[i];
		}
	}
}

//checked
__device__ void strip_mul(uint** strip, uint* Y, uint* out_block, uint strip_no, uint strip_end)
{
	uint tid; 
	uint id_row; //id_row decides the row index of output row
	uint id_col; //id_col decides the row index of input row
	
	uint *tempblock=out_block+blockIdx.x*BLOCK_SIZE;
	
	__shared__ uint block[BLOCK_SIZE];

	tid= threadIdx.x;
	while(tid<BLOCK_SIZE)
	{
		block[tid]=0;
		tid+=blockDim.x;
	}

	__syncthreads();
	tid = blockIdx.x*blockDim.x + threadIdx.x;

	while(tid<strip_end)
	{
		id_col=strip[strip_no][2*tid+1];
		id_row=strip[strip_no][2*tid];
		
		atomicXor((block+id_row%BLOCK_SIZE),Y[id_col]);//-arch sm_12	
		tid+=blockDim.x*gridDim.x;
	}
	__syncthreads();
	
	
	tid=threadIdx.x;
	while(tid<BLOCK_SIZE)
	{
		*(tempblock+tid)=block[tid];
		tid+=blockDim.x;
	}	
	//if(threadIdx.x+blockIdx.x==0)
		//cuPrintf("%u ",tempblock);
}
//checked
__global__ void global_strip_mul(uint** strip, uint* Y, uint* out_block, uint strip_no, uint strip_end)
{
	//reason: atomicXor() can be used only in device functions
	strip_mul(strip,Y,out_block,strip_no,strip_end);
}
uint* inp;
uint* out;
void* reduce(void*)
{
	for(uint i=0;i<BLOCK_SIZE;i++)
	{
		out[i]=0;
		for(uint j=0;j<NO_BLOCK;j++)
			out[i]^=inp[j*BLOCK_SIZE+i];
	}
	return NULL;
}

//checked
//void productBY(uint** dev_B, uint*dev_Y, uint* tempblock1, uint* tempblock2, uint* dev_temp1, uint* dev_temp2, uint* B_offsets, uint* BY, uint strip_no)
void productBY(uint** dev_B, uint*dev_Y, uint* B_offsets, uint* BY, uint strip_no)
{
	//pthread_t thread1,thread2;
	uint *tempblock1,*dev_temp1;
	uint *tempblock2,*dev_temp2;
	uint flag=cudaHostAllocMapped|cudaHostAllocPortable;
	cudaHostAlloc((void**)&tempblock1,NO_BLOCK*BLOCK_SIZE*sizeof(uint),flag);//FREE THIS MEMORY
	cudaHostAlloc((void**)&tempblock2,NO_BLOCK*BLOCK_SIZE*sizeof(uint),flag);
	cudaHostGetDevicePointer((void**)&dev_temp1,(void*)tempblock1,0);
	cudaHostGetDevicePointer((void**)&dev_temp2,(void*)tempblock2,0);
	uint switch_flag=1;
	for(uint j=0;j<strip_no;j++)
	{
		
		if(switch_flag)					
		{
			//cout<<j<<endl;	
			//if(j!=0)
				//pthread_join(thread2,NULL);
			memset(tempblock1,0,NO_BLOCK*BLOCK_SIZE*sizeof(uint));
			if(j==0)
				global_strip_mul<<<NO_BLOCK,512>>>(dev_B,dev_Y,dev_temp1,j,B_offsets[j]);
			else
				global_strip_mul<<<NO_BLOCK,512>>>(dev_B,dev_Y,dev_temp1,j,B_offsets[j]-B_offsets[j-1]);
			//cout<<cudaGetErrorString(cudaGetLastError())<<endl;
		}
		else
		{	//cout<<j<<endl;	
			//pthread_join(thread1,NULL);
			memset(tempblock2,0,NO_BLOCK*BLOCK_SIZE*sizeof(uint));
			global_strip_mul<<<NO_BLOCK,512>>>(dev_B,dev_Y,dev_temp2,j,B_offsets[j]-B_offsets[j-1]);
			//cout<<cudaGetErrorString(cudaGetLastError())<<endl;
		}
		
		if(switch_flag)
		{
			if(j!=0)
			{
				inp=tempblock2;
				out=(BY+j*BLOCK_SIZE);
				//pthread_create(&thread1,NULL,reduce,NULL);
				reduce(NULL);	
				//cout<<cudaGetErrorString(cudaGetLastError())<<endl;
			}
			switch_flag=0;
		}
		else
		{
			inp=tempblock1;
			out=(BY+j*BLOCK_SIZE);
			//pthread_create(&thread2,NULL,reduce,NULL);
			reduce(NULL);
			//cout<<cudaGetErrorString(cudaGetLastError())<<endl;
			switch_flag=1;
		}
	}
	cudaFreeHost(tempblock1);
	cudaFreeHost(tempblock2);
}



struct dense_mul_par
{
	uint* dense;
	uint densewords;
	uint no_cols;
	uint* out;
	uint* Y;
}*dense_par;

void* dense_mul(void*)
{
	uint j=0;
	uint row,col;	
	for(uint i=0;i<(dense_par->no_cols)*(dense_par->densewords);i++)
	{
		col=i/densewords;
		j=(N*i)%(dense_par->densewords*N);
		uint x=dense_par->dense[i];
		while(x)
		{
			row=j;
			if(x&1)
			{
				dense_par->out[row]^=dense_par->Y[col];
			}
			x=x>>1;
			j++;
		}		
	}
	return NULL;
}

//__global__ void find_block_end(uint** B, uint* Boffsets1, uint* Boffsets2, uint i,uint j,uint strip_end)
//{
//	uint k=Boffsets1[j];
//	while(k<strip_end && B[j][2*k+1]<(i+1)*BLOCK_SIZE)
//		k++;
//	Boffsets2[j]=k;
//}

//void find(uint** dev_B, uint* dev_Boffsets1, uint* dev_Boffsets2,uint strip_no,uint i,uint* Boffsets)
//{
//	for(uint j=0;j<strip_no;j++)
//		find_block_end<<<1,1>>>(dev_B,dev_Boffsets1, dev_Boffsets2,i,j,Boffsets[j]);
//}

__device__ void strip_trans_mul(uint** strip, uint* BY, uint* out_block, uint strip_no, uint strip_end)
{
	uint tid; 
	uint id_row; //id_row decides the row index of output row
	uint id_col; //id_col decides the row index of input row
	
	
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	while(tid<strip_end)
	{
		id_row=strip[strip_no][2*tid+1];
		id_col=strip[strip_no][2*tid];
	
		atomicXor((out_block+id_row),BY[id_col]);//-arch sm_12	
		tid+=blockDim.x*gridDim.x;
	}	
}

//checked
__global__ void global_trans_strip_mul(uint** strip, uint* BY, uint* out_block, uint strip_no, uint strip_end)
{
	//reason: atomicXor() can be used only in device functions
	strip_trans_mul(strip, BY, out_block, strip_no, strip_end);
}

void productV(uint** dev_B, uint* dev_BY, 
				uint* B_offsets, uint* dev_V0, 
				uint strip_no)
{
	for(uint j=0;j<strip_no;j++)
	{
			if (j==0)
				global_trans_strip_mul<<<NO_BLOCK,512>>>(dev_B,dev_BY,dev_V0,j,B_offsets[j]);
			else
				global_trans_strip_mul<<<NO_BLOCK,512>>>(dev_B,dev_BY,dev_V0,j,B_offsets[j]-B_offsets[j-1]);
	}
}

__device__  void dev_trans_dense_mul(uint no_rows, uint densewords, uint* dense, uint* out, uint* BY)
{
	uint j=0;
	uint row,col,tid;
	__shared__ uint sharedY[2048];//using a large constant
	
	tid=threadIdx.x;
	while(tid<densewords*N)
	{
		sharedY[tid]=BY[tid];
		tid+=blockDim.x;
	}
	__syncthreads();
	
	tid=blockIdx.x*blockDim.x+threadIdx.x;	
	while(tid<no_rows*densewords)
	{
		row=tid/densewords;
		j=(N*tid)%(densewords*N);
		uint x=dense[tid];
		while(x)
		{
			col=j;
			if(x&1)
			{
				atomicXor((out+row),sharedY[col]);
			}
			x=x>>1;
			j++;
		}
		tid+=gridDim.x*blockDim.x;		
	}
}

__global__ void trans_dense_mul(uint no_rows, uint densewords, uint* dense, uint* out, uint* BY)
{
	dev_trans_dense_mul(no_rows, densewords, dense, out, BY);
}

//for product BV
__device__  void device_dense_mul(uint no_cols, uint densewords, uint* dense, uint* BV, uint* V)
{
	uint j=0;
	uint row,col,tid;
	__shared__ uint sharedBV[2048];//using a large constant
	
	tid=threadIdx.x;
	while(tid<densewords*N)
	{
		sharedBV[tid]=0;
		tid+=blockDim.x;
	}
	__syncthreads();
	
	tid=blockIdx.x*blockDim.x+threadIdx.x;	
	while(tid<no_cols*densewords)
	{
		col=tid/densewords;
		j=(N*tid)%(densewords*N);
		uint x=dense[tid];
		while(x)
		{
			row=j;
			if(x&1)
			{
				atomicXor((sharedBV+row),V[col]);
			}
			x=x>>1;
			j++;
		}
		tid+=gridDim.x*blockDim.x;		
	}
	__syncthreads();
	
	tid=threadIdx.x;
	while(tid<densewords*N)
	{
		atomicXor((BV+tid),sharedBV[tid]);
		tid+=blockDim.x;
	}
}

__global__ void dev_dense_mul(uint no_cols, uint densewords, uint* dense, uint* BV, uint* V)
{
	device_dense_mul(no_cols, densewords, dense, BV, V);
}

struct host_mul_par
{
	uint* inp;
	uint* out;
	uint inp_size;
}*host_par;
void* host_mul_NnnN(void*)
{
	uint j;
	memset(out,0,sizeof(uint)*N);
	for(uint i=0;i<host_par->inp_size;i++)
	{
		uint x=host_par->inp[i];
		j=0;
		while(x)
		{
			if(x&1)
				host_par->out[j]^=host_par->inp[i];
			x=x>>1;
			j++;			
		}
	}
	return NULL;
}

__device__ void device_mul_NnnN(uint* u, uint* v, uint size, uint* out)
{
	uint tid;
	uint x,j;
	__shared__ uint block[N];
	//for(int i=0;i<10;i++)
		//cuPrintf("%u %u\n",u[i],v[i]);
	tid=threadIdx.x;
	while(tid<N)
	{
		block[tid]=0;
		tid+=blockDim.x;
	}
	__syncthreads();
	
	tid=blockIdx.x*blockDim.x+threadIdx.x;
	while(tid<size)
	{
		x=u[tid];
		//cuPrintf("%u\n",x);
		j=0;
		while(x)
		{
			if(x&1)
			{
				//cuPrintf("%u %u ",*(block+j),v[tid]);			
				atomicXor((block+j),v[tid]);
				//cuPrintf("%u\n",*(block+j));
			}
			x=x>>1;
			j++;			
		}
		tid+=gridDim.x*blockDim.x;
	}
	__syncthreads();
	
	tid=threadIdx.x;
	while(tid<N)
	{
		*(out+blockIdx.x*N+tid)=block[tid];
		tid+=blockDim.x;
	}
//	if(threadIdx.x+blockIdx.x==0)
//		for(int i=0;i<100;i++)
//			cuPrintf("%u ",out[i]);
//	cuPrintf("\n");	
	
}

__global__ void dev_mul_NnnN(uint* u, uint* v, uint size, uint* out)
{
	device_mul_NnnN(u, v, size, out);
}

void reduceVA2V(uint* inp, uint* out)
{
	for(uint i=0;i<N;i++)
	{
		out[i]=0;
		for(uint j=0;j<NO_BLOCK;j++)
			out[i]^=inp[j*N+i];
	}
}

__device__ void dev_mul_nNNN(uint* inpl, uint* inps, uint* out, uint size)
{
	__shared__ uint blockNN[N];
	uint tid;
	
	tid=threadIdx.x;
	while(tid<N)
	{
		blockNN[tid]=inps[tid];
		tid+=blockDim.x;
	}
	__syncthreads();
	
	uint x,j;
	tid=blockIdx.x*blockDim.x+threadIdx.x;
	while(tid<size)
	{
		x=inpl[tid];
		j=0;
		while(x)
		{
			if(x&1)
				atomicXor((out+tid),blockNN[j]);
			x=x>>1;
			j++;
		}	
		tid+=blockDim.x*gridDim.x;	
	}
}

__global__ void mul_nNNN(uint* inpl, uint* inps, uint* out, uint size)
{
	dev_mul_nNNN(inpl, inps, out, size);
}

void swap(uint x,uint y)
{
	uint temp=x;
	x=y;
	y=temp;
}

void load(uint* S,uint cj)
{
	uint x=1;
	x=x<<cj;
	*S=*S|x;
}

void inverse(uint *T,uint *S_1,uint *W_inv,uint* S)
{
	uint abc;
	uint M1[N],M2[N],c[N],t[N];
	memcpy(M1,T,sizeof(uint)*N);
	M2[31]=1;
	for(int i=30;i>=0;i--)
	{
		M2[i]=M2[i+1];
		M2[i]=M2[i]<<1;
	}
	*S=0;
	uint y=1;
	y=y<<31;
	//cout<<y<<endl;
	int a=0,p=0,flag=0,j=0;
	int count;
	//memset(c,55,sizeof(unsigned int)*N);//55 here is a random value to mark the columns in S_1
	for(int i=0;i<32;i++)
		c[i]=55;
	for(int i=0;i<N;i++)
	{
		flag=count=0;
		for(j=0;j<N;j++)
		{
			if(T[j]&y)
				count++;
			if(count>1) break;
		}
		if(count==1)
		{
			for(int k=0;k<32;k++)
			{
				if(S_1[k]&y)
				{
					flag++;
					t[p++]=i;
					break;
				}
			}
		}
		if(count>1||count==0||flag==32)
		{
			c[a++]=i;
		}
		y=y>>1;
	}

	for(int i=a,j=0;i<N;i++)
		c[i]=t[j++];
	
	y=1;
	y=y<<(31-j);

	for(j=0;j<N;j++)
	{
		uint k=j;
		uint x;
		x=M1[c[j]];
		x=x&y;
		while(x==0 && k<N)
		{
			uint p=M1[c[k]];
			p=p&y;

			if(p!=0)
			{
				swap(M1[c[k]],M1[c[j]]);
				swap(M2[c[k]],M2[c[j]]);
			}
			x=M1[c[j]];
			x=x&y;
			k++;
		}
		
		x=M1[c[j]];
		x=x&y;
		if(x!=0)
		{
			load(S,c[j]);
			for(uint i=0;i<N;i++)
			{
				if(i!=c[j])
				{
					uint p=M1[c[i]];
					p=p&y;
					if(p==1)
					{
						M1[c[i]]^=M1[c[j]];
						M2[c[i]]^=M2[c[j]];
					}
				}
			}
		}
		else
		{
			uint k=j;
			uint x;
			x=M2[c[j]];
			x=x&y;
			while(x==0 && k<N)
			{

				uint p=M2[c[k]];
				p=p&y;
				if(p!=0)
				{
					swap(M1[c[k]],M1[c[j]]);
					swap(M2[c[k]],M2[c[j]]);
		 		}
		 		k++;
		 		x=M2[c[j]];
				x=x&y;
		 	}
			x=M2[c[j]];
			x=x&y;
			if(x!=0)
				cout<<j<<" ";
			for(int i=0;i<N;i++)
			{
				if(i!=c[j])
				{
					uint p=M2[c[i]];
					p=p&y;
					if(p==1)
					{
						M2[c[i]]^=M2[c[j]];
						M1[c[i]]^=M1[c[j]];
					}
				}
			}
			M1[c[j]]=0;
			M2[c[j]]=0;
		}
		y=y>>1;

	}
	memcpy(W_inv,M2,sizeof(uint)*N);
}

struct D_i
{
	uint *Di;
	uint *Wi;
	uint *Zi;
}*D_par;

void* Di(void*)
{
	uint In[N];
	for(int i=0;i<N;i++)
	{
		In[i]=1<<(31-i);
	}
	mul_NNNN(D_par->Wi,D_par->Zi,D_par->Di);
	add_NNNN(D_par->Di,In,D_par->Di);
	return NULL;
}

struct E_i
{
	uint *Ei;
	uint *Wi_1;
	uint *SSi;
	uint *Ci;
}*E_par;

void* Ei(void*)
{
	uint product[N];
	mul_NNNN(E_par->Ci,E_par->SSi,product);
	mul_NNNN(E_par->Wi_1,product,E_par->Ei);
	return NULL;
}

struct F_i
{
	uint* Fi;
	uint* Wi_1;
	uint* Wi_2;
	uint* SSi;
	uint* Ci_1;
	uint* Zi_1;
}*F_par;
void* Fi(void*)
{
	uint In[N];
	uint prod1[N],prod2[N];
	for(int i=0;i<N;i++)
	{
		In[i]=1<<(31-i);
	}
	add_NNNN(F_par->Ci_1,In,F_par->Fi);
	add_NNNN(F_par->Fi,F_par->Wi_1,F_par->Fi);
	mul_NNNN(F_par->Wi_2,F_par->Fi,prod1);
	mul_NNNN(prod1,F_par->Zi_1,prod2);
	mul_NNNN(prod2,F_par->SSi,F_par->Fi);
	return NULL;
}
//n_N on Host Pinned Memory
//m_N on global memory of device
//by a_offsets we mean offsets for strips

int main(void)
{
	//flags to be set before any memory is allocated on GPU
	cudaSetDeviceFlags(cudaDeviceMapHost|cudaDeviceBlockingSync);
	
	//size_t freem;
	//size_t totalm;
	
	uint *B;
	uint **dev_B;
	uint **Bcopy;
	uint *B_offsets;
	read_sizes();
	
	uint *dense;
	uint *dev_dense;
	
	uint* Y;
	uint* dev_Y;
	
	uint* BY;
	uint* dev_BY;
	
	uint* dev_V0;
	
	uint* dev_AV;
	
	uint *Ci,*Ci_1;
	
	uint*VTA2V;
	
	uint *Wi, *Wi_1, *Wi_2;
	uint  *SSi, *SSi_1, S;
	uint *Vi, *Vi_1, *Vi_2, *Vi1;
	uint* X;
	uint *D,*E, *F;
	uint *productN1,*productN2;
	uint *Zi,*Zi_1;
	
	uint* dev_SSi;
	uint *dev_D, *dev_E, *dev_F;
	uint *dev_Wi, *dev_Wi_1, *dev_Wi_2;
	uint *dev_prodN2;
	
	uint* temp;
	pthread_t Ct,Dt,Et,Ft;
	pthread_t dense_t;
	uint strip_no=dims[0]/BLOCK_SIZE;
	
	B=(uint*)malloc(tell_size()*sizeof(uint));
	read_matrix(B,tell_size());
	cout<<"out of read_matrix\n";
		
	B_offsets=(uint*)malloc(strip_no*sizeof(uint));	
	read_matrix_offsets(B_offsets,(dims[0]/BLOCK_SIZE));//1 less than actual no of strips
	cout<<"out of matrix_offsets"<<endl;
				
	densewords=(dims[1]+N-1)/N;
	cout<<densewords<<endl;
	uint flag=cudaHostAllocPortable|cudaHostAllocMapped;
	cudaHostAlloc((void**)&dense,dims[2]*sizeof(uint),flag);
	cudaHostGetDevicePointer((void**)&dev_dense,(void*)dense,0);
	
	read_matrix3(dense,densewords*dims[2]);
	cout<<"reached in main\n";
	Bcopy=(uint**)malloc(strip_no*sizeof(uint*));
	
	cudaMalloc((void**)&dev_B,sizeof(uint*)*strip_no);
	cout<<"two D pointers allocated\n";
	
	uint size;
	uint offset=0;
	uint* Bcopyi;
	uint* Boffset;

	//copying indices to device memory from host to device strip by strip	
	for(int i=0;i<strip_no;i++)
	{
		if(i!=0)
			size=B_offsets[i]-B_offsets[i-1];
		else
			size=B_offsets[i];
		size*=2;//in matrix_read I stored k instead 2k+1
			
		cudaMalloc((void**)&Bcopy[i],sizeof(uint)*size);
		
		Bcopyi=Bcopy[i];
		Boffset=B+offset;
		
		cudaMemcpy(Bcopyi,
							Boffset,
							sizeof(uint)*size,
							cudaMemcpyHostToDevice);
		offset+=size;
	}
	
	cudaMemcpy(dev_B,Bcopy,sizeof(uint*)*strip_no,cudaMemcpyHostToDevice);
	
	cout<<"before free\n";
	free(B);
	cout<<"after free"<<endl;
	
	cudaHostAlloc((void**)&Y,dims[2]*sizeof(uint),flag);
	cudaHostGetDevicePointer((void**)&dev_Y,(void*)Y,0);
	
	cout<<"allocated for Y"<<endl;
	rand_gen(Y,dims[2]);
	
	cudaHostAlloc((void**)&BY,sizeof(uint)*dims[0],flag);
	cudaHostGetDevicePointer((void**)&dev_BY,(void*)BY,0);
	
	memset(BY,0,sizeof(uint)*dims[0]);
	
	//B*Y
	cout<<"k this is it\n";
	
	//dense parameters for the dense multipliction on host thread
	dense_par=(struct dense_mul_par*)malloc(sizeof(struct dense_mul_par));
	dense_par->dense=dense;
	dense_par->densewords=densewords;
	dense_par->no_cols=dims[2];
	dense_par->out=(uint*)malloc(sizeof(uint)*densewords*N);// no rows in output is densewords*N
	dense_par->Y=Y;
	memset(dense_par->out,0,sizeof(uint)*densewords*N);
	cout<<"hi\n";
	
	//dense multiplication on host thread
	pthread_create(&dense_t,NULL,dense_mul,NULL);
	cout<<"k till now\n";
	
	//sparse multplication on gpu
	productBY(dev_B, dev_Y, B_offsets, BY, strip_no);
	pthread_join(dense_t,NULL);
	
	//xor out to by
	for(uint i=0;i<densewords*N;i++)
		BY[i]^=dense_par->out[i];
		
	//move Y to normal host memory and reuse the memory in Y as V0
	temp=Y;
	cudaMalloc((void**)&dev_V0,sizeof(uint)*dims[2]);
	dev_Y=NULL;
	Y=(uint*)malloc(dims[2]*sizeof(uint));
	memcpy(Y,temp,dims[2]*sizeof(uint));
	//memset(V0,0,dims[2]*sizeof(uint));
	cudaFreeHost(temp);
	
	//dense multplication for BT*BY
	trans_dense_mul<<<NO_BLOCK,512>>>(dims[2], densewords, dev_dense, dev_V0, dev_BY);
	cout<<"product V will start\n";
	
	//sparse multiplication of BT*BY
	productV(dev_B, dev_BY, B_offsets, dev_V0, strip_no);
	
	//B*V
	
	//here we will reuse BY memory for BV0
	memset(BY,0,dims[0]*sizeof(uint));
	
	//dense multiplication of BV0
	cout<<"starting devdense\n";
	dev_dense_mul<<<NO_BLOCK,512>>>(dims[2], densewords, dev_dense, dev_BY, dev_V0);
	
	//sparse multiplication of BV0
	cout<<"starting BV0\n";

	productBY(dev_B, dev_V0, B_offsets, BY, strip_no);
	
	//allocating memory for AV
	cudaMalloc((void**)&dev_AV, sizeof(uint)*dims[2]);
	cudaMemset(dev_AV,0,sizeof(uint)*dims[2]);
	
	cout<<"starting product AV\n"<<flush;
	productV(dev_B, dev_BY, B_offsets, dev_AV, strip_no);
	
	trans_dense_mul<<<NO_BLOCK,512>>>(dims[2], densewords, dev_dense, dev_AV, dev_BY);
	
	//Ci=BVT * BV 
	uint abc;
	
	Ci=(uint*)malloc(sizeof(uint)*N);
	Ci_1=(uint*)malloc(sizeof(uint)*N);
	memset(Ci_1,0,sizeof(uint)*N);
	host_par=(struct host_mul_par*)malloc(sizeof(struct host_mul_par));
	host_par->inp=BY;
	host_par->out=Ci;
	host_par->inp_size=dims[0];
	
	host_mul_NnnN(NULL);
	
	//VTA2V product
	uint *tempblock3,*dev_temp3;
	cudaHostAlloc((void**)&tempblock3,N*NO_BLOCK*sizeof(uint),cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&dev_temp3,(void*)tempblock3,0);
	cout<<"before memset of tempblock1\n"<<flush;
	memset(tempblock3,0,sizeof(uint)*N*NO_BLOCK);
	
	VTA2V=(uint*)malloc(sizeof(uint)*N);

	//allocating memory for Wi,Wi_1,Wi_2
	cudaHostAlloc((void**)&Wi,sizeof(uint)*N,flag);
	cudaHostAlloc((void**)&Wi_1,sizeof(uint)*N,flag);
	cudaHostAlloc((void**)&Wi_2,sizeof(uint)*N,flag);
	
	cudaHostGetDevicePointer((void**)&dev_Wi,(void*)Wi,0);
	cudaHostGetDevicePointer((void**)&dev_Wi_1,(void*)Wi_1,0);
	cudaHostGetDevicePointer((void**)&dev_Wi_2,(void*)Wi_2,0);
	
	memset(Wi_1,0,sizeof(uint)*N);
	memset(Wi_2,0,sizeof(uint)*N);
		
	//allocating memoru for SSi,SSi_1
	cudaHostAlloc((void**)&SSi,sizeof(uint)*N,flag);
	cudaHostGetDevicePointer((void**)&dev_SSi,(void*)SSi,0);
	SSi_1=(uint*)malloc(sizeof(uint)*N);
	
	for(int i=0;i<N;i++)
		SSi_1[i]=1<<(31-i);
	//allocating Vi,Vi_1,Vi_2
	cudaMalloc((void**)&Vi,sizeof(uint)*dims[2]);
	cudaMemcpy(Vi,dev_V0,sizeof(uint)*dims[2],cudaMemcpyDeviceToDevice);
	
	cudaMalloc((void**)&Vi_1,sizeof(uint)*dims[2]);
	cudaMalloc((void**)&Vi_2,sizeof(uint)*dims[2]);

	cudaMemset(Vi_1,0,sizeof(uint)*dims[2]);
	cudaMemset(Vi_2,0,sizeof(uint)*dims[2]);
	
	cudaMalloc((void**)&X,sizeof(uint)*dims[2]);
	cudaMemset(X,0,sizeof(uint)*dims[2]);
	
	cudaHostAlloc((void**)&productN1,sizeof(uint)*N,cudaHostAllocMapped);
	cudaHostAlloc((void**)&productN2,sizeof(uint)*N,cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&dev_prodN2,(void*)productN2,0);
	
	Zi=(uint*)malloc(sizeof(uint)*N);
	Zi_1=(uint*)malloc(sizeof(uint)*N);
	memset(Zi_1,0,sizeof(uint)*N);
	
//	D=(uint*)malloc(sizeof(uint)*N);
//	E=(uint*)malloc(sizeof(uint)*N);
//	F=(uint*)malloc(sizeof(uint)*N);
	cudaHostAlloc((void**)&D,sizeof(uint)*N,flag);
	cudaHostAlloc((void**)&E,sizeof(uint)*N,flag);
	cudaHostAlloc((void**)&F,sizeof(uint)*N,flag);
	
	cudaHostGetDevicePointer((void**)&dev_D,(void*)D,0);
	cudaHostGetDevicePointer((void**)&dev_E,(void*)E,0);
	cudaHostGetDevicePointer((void**)&dev_F,(void*)F,0);
	
	D_par=(struct D_i*)malloc(sizeof(struct D_i));
	D_par->Wi=Wi;
	D_par->Zi=Zi;
	D_par->Di=D;
	
	E_par=(struct E_i*)malloc(sizeof(struct E_i));
	E_par->Wi_1=Wi_1;
	E_par->SSi=SSi;
	E_par->Ci=Ci;
	E_par->Ei=E;
	
	F_par=(struct F_i*)malloc(sizeof(struct F_i));
	F_par->Wi_2=Wi_2;
	F_par->Ci_1=Ci_1;
	F_par->Wi_1=Wi_1;
	F_par->Zi_1=Zi_1;
	F_par->SSi=SSi;
	F_par->Fi=F;

	cudaMalloc((void**)&Vi1,dims[2]*sizeof(uint));
	
	//ITERATION STARTS HERE
	
	//
	//
	cudaEvent_t start;
	cudaEvent_t stop;
	float ms;
	cudaEventCreate(&start,0);
	cudaEventCreate(&stop,0);
	cudaEventRecord(start,0);
	cudaEventSynchronize(start);
	//for(int abc=0;abc<5;abc++)
	{
	//VTA2V
	dev_mul_NnnN<<<NO_BLOCK,512>>>(dev_AV,dev_AV,dims[2],dev_temp3);
	reduceVA2V(tempblock3,VTA2V);
	//pthread_join(BV_t,NULL);
	cout<<"VTA2V\n";
	
	
	// find W_inv and S
	inverse(Ci,SSi_1,Wi,&S);
	cout<<"inverse done\n"<<flush;

	for(int i=0;i<N;i++)
	{
		uint x=1<<(31-i);
		if(S&x)
		{
			SSi[i]=x;
		}
	}
	cout<<"starting X\n";
	// Find X
	//ViT*V0
	dev_mul_NnnN<<<NO_BLOCK,512>>>(Vi,dev_V0,dims[2],dev_temp3);
	//cudaThreadSynchronize();
	reduceVA2V(tempblock3,productN1);
	
	//W_inv*(ViT*V0)
	mul_NNNN(productN1,Wi,productN2);
	//X=X+Vi*(W_inv*()ViT*V0))
	
	cout<<"calculating X\n"<<flush;
	mul_nNNN<<<NO_BLOCK,512>>>(Vi,dev_prodN2,X,dims[2]);
	cout<<"X done\n";
	
	//Zi 
	mul_NNNN(VTA2V,SSi,productN1);
	add_NNNN(productN1,Ci,Zi);
	
	//Di+1
	pthread_create(&Dt,NULL,Di,NULL);
	//Di(NULL);
	
	//Ei+1
	pthread_create(&Et,NULL,Ei,NULL);
	//Ei(NULL);
	
	//Fi+1
	pthread_create(&Ft,NULL,Fi,NULL);
	//Fi(NULL);
	
	pthread_join(Dt,NULL);
	pthread_join(Et,NULL);
	pthread_join(Ft,NULL);
	
	//Calculation  of Vi+1
	cudaMemset(Vi1,0,dims[2]*sizeof(uint));
	mul_nNNN<<<NO_BLOCK,512>>>(dev_AV,dev_SSi,Vi1,dims[2]);
	mul_nNNN<<<NO_BLOCK,512>>>(Vi,dev_D,Vi1,dims[2]);
	mul_nNNN<<<NO_BLOCK,512>>>(Vi,dev_E,Vi1,dims[2]);
	mul_nNNN<<<NO_BLOCK,512>>>(Vi,dev_F,Vi1,dims[2]);
	cout<<cudaGetErrorString(cudaGetLastError());
	//BVi+1
	
	productBY(dev_B, Vi1, B_offsets, BY, strip_no);
	dev_dense_mul<<<NO_BLOCK,512>>>(dims[2], densewords, dev_dense, dev_BY, Vi1);
	
	//AVi+1
	pthread_create(&Ct,NULL,host_mul_NnnN,NULL);
	productV(dev_B, dev_BY, B_offsets, dev_AV, strip_no);
	trans_dense_mul<<<NO_BLOCK,512>>>(dims[2], densewords, dev_dense, dev_AV, dev_BY);
	
	//Ci+1
	memcpy(Ci_1,Ci,sizeof(uint)*N);
	//host_mul_NnnN(NULL);
	pthread_join(Ct,NULL);
	for(uint i=0;i<N;i++)
		cout<<Ci[i]<<" ";
	cout<<endl;
	
	//reordering pointers
	//Vis
	temp=Vi_2;
	Vi_2=Vi_1;
	Vi_1=Vi;
	Vi=Vi1;
	Vi1=temp;
	
	//Wis
	temp=Wi_2;
	Wi_2=Wi_1;
	Wi_1=Wi;
	Wi=temp;
	
	//devWis
	temp=dev_Wi_2;
	dev_Wi_2=dev_Wi_1;
	dev_Wi_1=dev_Wi;
	dev_Wi=temp;
	
	//SSis
	memcpy(SSi_1,SSi,N*sizeof(uint));
	
	//Zi
	temp=Zi_1;
	Zi_1=Zi;
	Zi=temp;
}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms,start,stop);
	cout<<ms<<endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
cout<<"reached here\n";
	free(B_offsets);
	for(int i=0;i<strip_no;i++)
		cudaFree(Bcopy[i]);
	free(Bcopy);
	cout<<"ok ,,,\n";
	cudaFree(dev_B);
	cudaFreeHost(dense);
	cout<<"ok ,,,\n";
	cudaFreeHost(Y);
	cudaFreeHost(BY);
	cout<<"ok ,,,\n";
	cudaFree(dev_V0);
	cout<<"ok ,,,\n";
	free(Ci);
	free(Ci_1);
	free(VTA2V);
	cout<<"ok ,,,\n";
	cudaFreeHost(Wi);
	cudaFreeHost(Wi_1);
	cudaFreeHost(Wi_2);
	cout<<"ok ,,,\n";
	cudaFreeHost(SSi);
	free(SSi_1);
	cout<<"ok ,,,\n";
	cudaFree(Vi);
	cudaFree(Vi_1);
	cudaFree(Vi_2);
	cout<<"ok ,,,\n";
	cudaFreeHost(D);
	cudaFreeHost(E);
	cudaFreeHost(F);
	cout<<"ok ,,,\n";
	cudaFreeHost(productN1);
	cudaFreeHost(productN2);
	cout<<"ok ,,,\n";
	free(Zi);
	free(Zi_1);
	
	cout<<"abc:\n";
	return 0;
}
	
