// Faithful SpMV A/B on cf's real 990k pitzDaily matrix (same mesh -> same gather pattern).
// Compares: (1) cf's amulKernel (LDU losort gather, 1-thread/cell), (2) CSR vector-per-row,
// (3) cuSPARSE cusparseSpMV. FP64. Same matrix, verified identical result.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>
using std::vector;
#define CK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define SP(x) do{auto e=(x); if(e!=CUSPARSE_STATUS_SUCCESS){printf("cusparse err %s:%d %d\n",__FILE__,__LINE__,(int)e);exit(1);} }while(0)

static vector<int> readList(const char* fn){
    std::ifstream f(fn); std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t lp=s.find('(');                       // first '(' = data list opener (FoamFile uses {})
    size_t e=s.find_last_not_of(" \t\n\r", lp-1);
    size_t b=s.find_last_not_of("0123456789", e);
    int N=atoi(s.substr(b+1, e-b).c_str());
    std::istringstream is(s.substr(lp+1));
    vector<int> v(N); for(int i=0;i<N;i++) is>>v[i]; return v;
}

// cf's exact fine-level SpMV (device_spmv.cu amulKernel)
__global__ void amulKernel(int nC,const double*__restrict__ diag,const double*__restrict__ upper,const double*__restrict__ lower,
    const int*__restrict__ nei,const int*__restrict__ owner,const int*__restrict__ ownerStart,
    const int*__restrict__ losort,const int*__restrict__ losortStart,const double*__restrict__ psi,double*__restrict__ Apsi){
    const int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nC) return;
    double s=diag[c]*psi[c];
    for(int f=ownerStart[c];f<ownerStart[c+1];++f) s+=upper[f]*psi[nei[f]];
    for(int k=losortStart[c];k<losortStart[c+1];++k){const int f=losort[k]; s+=lower[f]*psi[owner[f]];}
    Apsi[c]=s;
}
// CSR vector-per-row: TPR threads cooperate on one row (TPR=8 fits FV degree ~5-7)
template<int TPR> __global__ void csrVecKernel(int nR,const int*__restrict__ rp,const int*__restrict__ ci,
    const double*__restrict__ v,const double*__restrict__ x,double*__restrict__ y){
    const int tid=blockIdx.x*blockDim.x+threadIdx.x; const int row=tid/TPR; const int lane=tid%TPR;
    if(row>=nR) return; double s=0;
    for(int k=rp[row]+lane;k<rp[row+1];k+=TPR) s+=v[k]*x[ci[k]];
    for(int o=TPR/2;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o,TPR);
    if(lane==0) y[row]=s;
}

int main(int argc,char**argv){
    const char* mesh=argc>1?argv[1]:"/tmp/cfprof_M9/constant/polyMesh";
    std::string of=std::string(mesh)+"/owner", nf=std::string(mesh)+"/neighbour";
    vector<int> owner=readList(of.c_str()), nei=readList(nf.c_str());
    int nF=nei.size(); owner.resize(nF);         // internal faces only
    int nC=0; for(int f=0;f<nF;f++){nC=std::max(nC,std::max(owner[f],nei[f])+1);}
    printf("matrix: nCells=%d nFaces=%d nnz=%d avgDeg=%.2f\n",nC,nF,nC+2*nF,(double)(nC+2*nF)/nC);
    // LDU build (ownerStart, losort, losortStart), cf's buildDeviceLdu
    vector<int> ownerStart(nC+1,0), losortStart(nC+1,0);
    for(int f=0;f<nF;f++){ ownerStart[owner[f]+1]++; losortStart[nei[f]+1]++; }
    for(int c=0;c<nC;c++){ ownerStart[c+1]+=ownerStart[c]; losortStart[c+1]+=losortStart[c]; }
    vector<int> losort(nF); { vector<int> t(losortStart.begin(),losortStart.end()); for(int f=0;f<nF;f++) losort[t[nei[f]]++]=f; }
    // SPD coefficients (values don't affect the memory pattern; diag dominant)
    vector<double> diag(nC), upper(nF,-1.0), lower(nF,-1.0);
    for(int c=0;c<nC;c++) diag[c]=1.0+(ownerStart[c+1]-ownerStart[c])+(losortStart[c+1]-losortStart[c]);
    // CSR (merged, col-sorted) for the CSR/cuSPARSE paths
    int nnz=nC+2*nF; vector<int> rp(nC+1,0), ci(nnz); vector<double> cv(nnz);
    { vector<vector<std::pair<int,double>>> rows(nC);
      for(int c=0;c<nC;c++) rows[c].push_back({c,diag[c]});
      for(int f=0;f<nF;f++){ rows[owner[f]].push_back({nei[f],upper[f]}); rows[nei[f]].push_back({owner[f],lower[f]}); }
      int p=0; for(int c=0;c<nC;c++){ std::sort(rows[c].begin(),rows[c].end()); rp[c]=p; for(auto&e:rows[c]){ci[p]=e.first;cv[p]=e.second;p++;} } rp[nC]=p; }
    // device
    auto D=[&](const vector<double>&h){double*d;CK(cudaMalloc(&d,h.size()*8));CK(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice));return d;};
    auto I=[&](const vector<int>&h){int*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    double *d_diag=D(diag),*d_up=D(upper),*d_lo=D(lower),*d_cv=D(cv);
    int *d_owner=I(owner),*d_nei=I(nei),*d_os=I(ownerStart),*d_ls=I(losort),*d_lss=I(losortStart),*d_rp=I(rp),*d_ci=I(ci);
    vector<double> hx(nC); for(int c=0;c<nC;c++) hx[c]=1.0+0.001*(c%97);
    double *d_x=D(hx),*d_y1,*d_y2,*d_y3; CK(cudaMalloc(&d_y1,nC*8));CK(cudaMalloc(&d_y2,nC*8));CK(cudaMalloc(&d_y3,nC*8));
    const int TPB=256; auto tim=[&](const char*name,auto fn,double*yout){
        for(int i=0;i<5;i++) fn(); CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b); const int R=200;
        cudaEventRecord(a); for(int i=0;i<R;i++) fn(); cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms,a,b); double us=ms*1000.0/R;
        double gb=(double)(nnz*(8.0+4.0)+nC*(8.0+8.0+4.0))/1e9;   // rough SpMV byte model
        printf("  %-22s %7.1f us   %6.1f GB/s\n",name,us,gb/(us*1e-6));
    };
    tim("cf amulKernel (LDU)", [&]{ amulKernel<<<(nC+TPB-1)/TPB,TPB>>>(nC,d_diag,d_up,d_lo,d_nei,d_owner,d_os,d_ls,d_lss,d_x,d_y1); }, d_y1);
    tim("CSR vector-per-row(8)",[&]{ csrVecKernel<8><<<(nC*8+TPB-1)/TPB,TPB>>>(nC,d_rp,d_ci,d_cv,d_x,d_y2); }, d_y2);
    // cuSPARSE
    cusparseHandle_t h; SP(cusparseCreate(&h));
    cusparseSpMatDescr_t A; SP(cusparseCreateCsr(&A,nC,nC,nnz,d_rp,d_ci,d_cv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    cusparseDnVecDescr_t X,Y; SP(cusparseCreateDnVec(&X,nC,d_x,CUDA_R_64F)); SP(cusparseCreateDnVec(&Y,nC,d_y3,CUDA_R_64F));
    double al=1,be=0; size_t bs; SP(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&al,A,X,&be,Y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,&bs));
    void*buf; CK(cudaMalloc(&buf,bs)); SP(cusparseSpMV_preprocess(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&al,A,X,&be,Y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,buf));
    tim("cuSPARSE CSR ALG2",  [&]{ cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&al,A,X,&be,Y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,buf); }, d_y3);
    // verify
    vector<double> y1(nC),y2(nC),y3(nC);
    CK(cudaMemcpy(y1.data(),d_y1,nC*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y2.data(),d_y2,nC*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y3.data(),d_y3,nC*8,cudaMemcpyDeviceToHost));
    double e2=0,e3=0,nrm=0; for(int c=0;c<nC;c++){nrm+=y1[c]*y1[c];e2+=(y1[c]-y2[c])*(y1[c]-y2[c]);e3+=(y1[c]-y3[c])*(y1[c]-y3[c]);}
    printf("  verify: CSRvec vs LDU rel=%.2e   cuSPARSE vs LDU rel=%.2e\n",sqrt(e2/nrm),sqrt(e3/nrm));
    return 0;
}
