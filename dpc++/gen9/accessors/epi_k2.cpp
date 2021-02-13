// DPCPP TR

#define SYCL_SIMPLE_SWIZZLES
#define SYCL_INTEL_group_algorithms	
#include <CL/sycl.hpp>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include <float.h>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <omp.h>

#define MAX_WG_SIZE 256
#define SUBSLICES 3

float gammafunction(uint n)
{   
    if(n == 0)
        return 0.0f;
    float x = (n + 0.5f) * cl::sycl::log((float) n) - (n - 1) * cl::sycl::log(cl::sycl::exp((float) 1.0f));
    return x;
}

void generate_data(int long long N, int long long M, uint8_t **A, uint8_t **B)
{
    int i, j;

    srand(100);
    *A = (uint8_t *) _mm_malloc(N * M * sizeof(uint8_t), 64);
    *B = (uint8_t *) _mm_malloc(N * sizeof(uint8_t), 64);

    //Generate SNPs
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            //Generate Between 0 and 2
            (*A)[i * M + j] = rand() % 3;
        }
    }

    // generate phenotype
    for(i = 0; i < N; i++)
    {
        //Generate Between 0 and 1
        (*B)[i] = rand() % 2;
    }
}

uint8_t * transpose_data(int long long N, int long long M, uint8_t * A)
{
    int i, j;
    uint8_t *A_trans = (uint8_t *) _mm_malloc(M * N * sizeof(uint8_t), 64);
    
    for (i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            A_trans[j * N + i] = A[i * M + j];
        }
    }

    return A_trans;
}

void transposed_to_binary(uint8_t* original, uint8_t* original_ph, uint32_t** data_zeros, uint32_t** data_ones, int* phen_ones, int num_snp, int num_pac)
{
    int PP = ceil((1.0*num_pac)/32.0);
    uint32_t temp;
    int i, j, x_zeros, x_ones, n_zeros, n_ones;

    (*phen_ones) = 0;

    for(i = 0; i < num_pac; i++){
        if(original_ph[i] == 1){
            (*phen_ones) ++;
        }
    }

    int PP_ones = ceil((1.0*(*phen_ones))/32.0);
    int PP_zeros = ceil((1.0*(num_pac - (*phen_ones)))/32.0);

    // allocate data
    *data_zeros = (uint32_t*) _mm_malloc(num_snp*PP_zeros*2*sizeof(uint32_t), 64);
    *data_ones = (uint32_t*) _mm_malloc(num_snp*PP_ones*2*sizeof(uint32_t), 64);
    memset((*data_zeros), 0, num_snp*PP_zeros*2*sizeof(uint32_t));
    memset((*data_ones), 0, num_snp*PP_ones*2*sizeof(uint32_t));

    for(i = 0; i < num_snp; i++)
    {
        x_zeros = -1;
        x_ones = -1;
        n_zeros = 0;
        n_ones = 0;

        for(j = 0; j < num_pac; j++){
            temp = (uint32_t) original[i * num_pac + j];

            if(original_ph[j] == 1){
                if(n_ones%32 == 0){
                    x_ones ++;
                }
                // apply 1 shift left to 2 components
                (*data_ones)[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
                (*data_ones)[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
                // insert '1' in correct component
                if(temp == 0 || temp == 1){
                    (*data_ones)[i * PP_ones * 2 + x_ones*2 + temp ] |= 1;
                }
                n_ones ++;
            }else{
                if(n_zeros%32 == 0){
                    x_zeros ++;
                }
                // apply 1 shift left to 2 components
                (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
                (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
                // insert '1' in correct component
                if(temp == 0 || temp == 1){
                    (*data_zeros)[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
                }
                n_zeros ++;
            }
        }
    }
}

uint32_t* bin_to_trans(uint32_t* A, int num_snp, int PP)
{
    int i, j, S;
    uint32_t* A_trans = (uint32_t*) _mm_malloc(num_snp * PP * 2 * sizeof(uint32_t), 64);
    
    S = num_snp * PP;
    for(i = 0; i < num_snp; i++)
    {
        for(j = 0; j < PP; j++)
        {
            A_trans[(j * num_snp + i) * 2 + 0] = A[(i * PP + j) * 2 + 0];
            A_trans[(j * num_snp + i) * 2 + 1] = A[(i * PP + j) * 2 + 1];
        }
    }

    _mm_free(A);
    return A_trans;
}

int main(int argc, char **argv)
{
    int num_snp, num_pac, phen_ones, dim_epi, block_snp, x;
    uint8_t *SNP_Data, *Ph_Data;
    uint32_t *bin_data_ones, *bin_data_zeros;

    num_pac = atoi(argv[1]);
    num_snp = atoi(argv[2]);
    dim_epi = 2;
    block_snp = 64;

    // generate data set
    generate_data(num_pac, num_snp, &SNP_Data, &Ph_Data);
    SNP_Data = transpose_data(num_pac, num_snp, SNP_Data);
    transposed_to_binary(SNP_Data, Ph_Data, &bin_data_zeros, &bin_data_ones, &phen_ones, num_snp, num_pac);

    // get data set parameters
    int PP = ceil(1.0 * num_pac / 32.0);
    int PP_ones = ceil((1.0 * phen_ones)/32.0);
    int PP_zeros = ceil((1.0 * (num_pac - phen_ones)) / 32.0);
    int comb = (int) pow(3.0, dim_epi);

    uint32_t mask_zeros, mask_ones;
    mask_zeros = 0xFFFFFFFF;
    for(x = num_pac - phen_ones; x < PP_zeros * 32; x++)
        mask_zeros = mask_zeros >> 1;
    mask_ones = 0xFFFFFFFF;
    for(x = phen_ones; x < PP_ones * 32; x++)
        mask_ones = mask_ones >> 1;

    // transform data structures
    bin_data_ones = bin_to_trans(bin_data_ones, num_snp, PP_ones);
    bin_data_zeros = bin_to_trans(bin_data_zeros, num_snp, PP_zeros);
    _mm_free(SNP_Data);
    _mm_free(Ph_Data);

    // create DPC++ command queue (enable profiling and use GPU as device)
    auto property_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling(), cl::sycl::property::queue::in_order()};
    auto device_selector = cl::sycl::gpu_selector{};
    cl::sycl::queue queue(device_selector, property_list);

    // get device and context
    auto device = queue.get_device();
    auto context = queue.get_context();
    std::cout << "Selected device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
    
    // create buffers
    float *scores = (float*) _mm_malloc(num_snp * num_snp * sizeof(float), 64);
    for(x = 0; x < num_snp * num_snp; x++)
        scores[x] = FLT_MAX;
    cl::sycl::buffer<uint, 1> buf_dev_data_zeros(bin_data_zeros, cl::sycl::range<1>{static_cast<size_t>(num_snp * PP_zeros * 2)});
    cl::sycl::buffer<uint, 1> buf_dev_data_ones(bin_data_ones, cl::sycl::range<1>{static_cast<size_t>(num_snp * PP_ones  * 2)});
    cl::sycl::buffer<uint, 1> buf_dev_solutions(cl::sycl::range<1>{static_cast<size_t>(SUBSLICES)});
    cl::sycl::buffer<float, 1> buf_dev_scores(scores, cl::sycl::range<1>{static_cast<size_t>(num_snp * num_snp)});

    // setup kernel ND-range
    int num_snp_m = num_snp;
    while(num_snp_m % block_snp != 0)
        num_snp_m++;
    cl::sycl::range<2> global_epi(num_snp_m, num_snp_m);
    cl::sycl::range<2> local_epi(1, block_snp);
    cl::sycl::range<1> global_red(MAX_WG_SIZE * SUBSLICES);
    cl::sycl::range<1> local_red(MAX_WG_SIZE);

    // DPC++ kernel call
    queue.wait();
    double start = omp_get_wtime();
    // epistasis detection kernel
    queue.submit([&](cl::sycl::handler& h)
    {
        auto dev_data_zeros = buf_dev_data_zeros.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>(h);
        auto dev_data_ones = buf_dev_data_ones.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::constant_buffer>(h);
        auto dev_scores = buf_dev_scores.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>(h);
        h.parallel_for<class kernel_epi>(cl::sycl::nd_range<2>(global_epi, local_epi), [=](cl::sycl::nd_item<2> id)
        {
            int i, j, tid, p, k;
            float score = FLT_MAX;

            i = id.get_global_id(0);
            j = id.get_global_id(1);
            tid = i * num_snp + j;

            if(j > i && i < num_snp && j < num_snp)
            {
                uint ft[2 * 9];
                for(k = 0; k < 2 * 9; k++)
                    ft[k] = 0;
                
                uint t00, t01, t02, t10, t11, t12, t20, t21, t22;
                uint di2, dj2;
                uint* SNPi;
                uint* SNPj;

                // Phenotype 0
                SNPi = (uint*) &dev_data_zeros[i * 2];
                SNPj = (uint*) &dev_data_zeros[j * 2];
                for(p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp)
                {
                    di2 = ~(SNPi[p] | SNPi[p + 1]);
                    dj2 = ~(SNPj[p] | SNPj[p + 1]);

                    t00 = SNPi[p] & SNPj[p];
                    t01 = SNPi[p] & SNPj[p + 1];
                    t02 = SNPi[p] & dj2;
                    t10 = SNPi[p + 1] & SNPj[p];
                    t11 = SNPi[p + 1] & SNPj[p + 1];
                    t12 = SNPi[p + 1] & dj2;
                    t20 = di2 & SNPj[p];
                    t21 = di2 & SNPj[p + 1];
                    t22 = di2 & dj2;
                    
                    ft[0] += cl::sycl::popcount(t00);
                    ft[1] += cl::sycl::popcount(t01);
                    ft[2] += cl::sycl::popcount(t02);
                    ft[3] += cl::sycl::popcount(t10);
                    ft[4] += cl::sycl::popcount(t11);
                    ft[5] += cl::sycl::popcount(t12);
                    ft[6] += cl::sycl::popcount(t20);
                    ft[7] += cl::sycl::popcount(t21);
                    ft[8] += cl::sycl::popcount(t22);
                }
                // remainder
                p = 2 * PP_zeros * num_snp - 2 * num_snp;
                di2 = ~(SNPi[p] | SNPi[p + 1]);
                dj2 = ~(SNPj[p] | SNPj[p + 1]);
                di2 = di2 & mask_zeros;
                dj2 = dj2 & mask_zeros;

                t00 = SNPi[p] & SNPj[p];
                t01 = SNPi[p] & SNPj[p + 1];
                t02 = SNPi[p] & dj2;
                t10 = SNPi[p + 1] & SNPj[p];
                t11 = SNPi[p + 1] & SNPj[p + 1];
                t12 = SNPi[p + 1] & dj2;
                t20 = di2 & SNPj[p];
                t21 = di2 & SNPj[p + 1];
                t22 = di2 & dj2;

                ft[0] += cl::sycl::popcount(t00);
                ft[1] += cl::sycl::popcount(t01);
                ft[2] += cl::sycl::popcount(t02);
                ft[3] += cl::sycl::popcount(t10);
                ft[4] += cl::sycl::popcount(t11);
                ft[5] += cl::sycl::popcount(t12);
                ft[6] += cl::sycl::popcount(t20);
                ft[7] += cl::sycl::popcount(t21);
                ft[8] += cl::sycl::popcount(t22);

                // Phenotype 1
                SNPi = (uint*) &dev_data_ones[i * 2];
                SNPj = (uint*) &dev_data_ones[j * 2];
                for(p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp)
                {
                    di2 = ~(SNPi[p] | SNPi[p + 1]);
                    dj2 = ~(SNPj[p] | SNPj[p + 1]);

                    t00 = SNPi[p] & SNPj[p];
                    t01 = SNPi[p] & SNPj[p + 1];
                    t02 = SNPi[p] & dj2;
                    t10 = SNPi[p + 1] & SNPj[p];
                    t11 = SNPi[p + 1] & SNPj[p + 1];
                    t12 = SNPi[p + 1] & dj2;
                    t20 = di2 & SNPj[p];
                    t21 = di2 & SNPj[p + 1];
                    t22 = di2 & dj2;
                    
                    ft[9]  += cl::sycl::popcount(t00);
                    ft[10] += cl::sycl::popcount(t01);
                    ft[11] += cl::sycl::popcount(t02);
                    ft[12] += cl::sycl::popcount(t10);
                    ft[13] += cl::sycl::popcount(t11);
                    ft[14] += cl::sycl::popcount(t12);
                    ft[15] += cl::sycl::popcount(t20);
                    ft[16] += cl::sycl::popcount(t21);
                    ft[17] += cl::sycl::popcount(t22);
                }
                p = 2 * PP_ones * num_snp - 2 * num_snp;
                di2 = ~(SNPi[p] | SNPi[p + 1]);
                dj2 = ~(SNPj[p] | SNPj[p + 1]);
                di2 = di2 & mask_ones;
                dj2 = dj2 & mask_ones;

                t00 = SNPi[p] & SNPj[p];
                t01 = SNPi[p] & SNPj[p + 1];
                t02 = SNPi[p] & dj2;
                t10 = SNPi[p + 1] & SNPj[p];
                t11 = SNPi[p + 1] & SNPj[p + 1];
                t12 = SNPi[p + 1] & dj2;
                t20 = di2 & SNPj[p];
                t21 = di2 & SNPj[p + 1];
                t22 = di2 & dj2;

                ft[9]  += cl::sycl::popcount(t00);
                ft[10] += cl::sycl::popcount(t01);
                ft[11] += cl::sycl::popcount(t02);
                ft[12] += cl::sycl::popcount(t10);
                ft[13] += cl::sycl::popcount(t11);
                ft[14] += cl::sycl::popcount(t12);
                ft[15] += cl::sycl::popcount(t20);
                ft[16] += cl::sycl::popcount(t21);
                ft[17] += cl::sycl::popcount(t22);

                // compute score
                score = 0.0f;
                for(k = 0; k < 9; k++)
                    score += gammafunction(ft[k] + ft[9 + k] + 1) - gammafunction(ft[k]) - gammafunction(ft[9 + k]);
                score = cl::sycl::fabs((float) score);
                if(score == 0.0f)
                    score = FLT_MAX;
                dev_scores[tid] = score;
            }
            // end kernel
        });
    });
    
    // reduction kernel
    queue.submit([&](cl::sycl::handler& h)
    {
        auto dev_solutions = buf_dev_solutions.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>(h);
        auto dev_scores = buf_dev_scores.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>(h);
        h.parallel_for<class kernel_red>(cl::sycl::nd_range<1>(global_red, local_red), [=](cl::sycl::nd_item<1> id)
        {
            cl::sycl::group<1> gr = id.get_group();
            int size = num_snp * num_snp;
            int lid = id.get_local_id(0);
            int grid = gr.get_id(0);
            int gid = id.get_global_id(0);
            int gsize = MAX_WG_SIZE * SUBSLICES;
            int i, solution;
            float a, b;

            a = dev_scores[gid];
            solution = gid;
            for(i = gid + gsize; i < size; i += gsize)
            {
                b = dev_scores[i];
                if(b < a)
                {
                    a = b;
                    solution = i;
                }
            }
            b = a;
            id.barrier(cl::sycl::access::fence_space::local_space);
            a = sycl::ONEAPI::reduce<cl::sycl::group<1>, float, sycl::ONEAPI::minimum<float>>(gr, a, sycl::ONEAPI::minimum<float>());
            id.barrier(cl::sycl::access::fence_space::local_space);
            if(a == b)
            {
                dev_scores[grid] = a;
                dev_solutions[grid] = solution;
            }
            // end kernel
        });
    });
    queue.wait();

    // get host access
    auto dev_scores = buf_dev_scores.get_access<cl::sycl::access::mode::read>();
    auto dev_solutions = buf_dev_solutions.get_access<cl::sycl::access::mode::read>();
    
    // print results
    float score = FLT_MAX;
    uint32_t solution;
    for(x = 0; x < SUBSLICES; x++)
    {
        if(dev_scores[x] < score)
        {
            score = dev_scores[x];
            solution = dev_solutions[x];
        }
    }
    double end = omp_get_wtime();
    std::cout << "Score: " << score << std::endl;
    std::cout << "Solution: " << solution / num_snp << ", " << solution % num_snp << std::endl;
    std::cout << "Time: " << end - start << std::endl;

    // end
    _mm_free(bin_data_zeros);
    _mm_free(bin_data_ones);
    _mm_free(scores);
    return 0;
}
