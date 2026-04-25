/**
 * naive_attention.cu — CUDA implementation of Naive Causal Self-Attention.
 *
 * Formula:  O = softmax(causal_mask(Q @ K^T / sqrt(d))) @ V
 * Input:    Q, K, V, shape [B, H, S, D], half (fp16)
 * Output:   O, shape [B, H, S, D], half (fp16)
 *
 * 3 kernels: naive_gemm (QK^T + scale + mask) -> naive_softmax -> naive_pv (P @ V)
 * Grid: B*H blocks, S threads per block (one thread per row).
 *
 * Kernel code identical to csrc/naive_attention.cu.
 * Structure matches ascendc/naive_attention.asc for side-by-side comparison.
 */

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

// ============================================================================
// Config
// ============================================================================

// Debug shape: B=1, H=1, S=8, D=8
constexpr int DEBUG_B = 1;
constexpr int DEBUG_H = 1;
constexpr int DEBUG_S = 8;
constexpr int DEBUG_D = 8;

// Perf shape: B=2, H=4, S=256, D=64
constexpr int PERF_B = 2;
constexpr int PERF_H = 4;
constexpr int PERF_S = 256;
constexpr int PERF_D = 64;

// ============================================================================
// Kernel 1: naive_gemm — identical to csrc/naive_attention.cu
// ============================================================================

// gemm: aA@B + bC, A (m, k), B(n, k), C(m, n)
// m: seq_len, n: seq_len, k: head_dim
template <typename T>
__global__ void naive_gemm(
    const T* A,
    const T* B,
    T* C,
    T a,
    T b,
    unsigned int M,
    unsigned int N,
    unsigned int K
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            C[idx * N + j] = -INFINITY;
        }
        else {
            T sum = static_cast<T>(0.f);
            for (int k = 0; k < K; k++) {
                sum += A[idx * K + k] * B[(blockDim.x * blockIdx.x + j) * K + k];
            }
            C[idx * N + j] = a * sum + b * C[idx * N + j];
        }
    }
}

// ============================================================================
// Kernel 2: naive_softmax — identical to csrc/naive_attention.cu
// ============================================================================

// softmax: exp(x - max(x)) / sum(exp(x - max(x)))
template <typename T>
__global__ void naive_softmax(
    T* input,
    T* output,
    unsigned int N
)
{
    // 3-pass softmax
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    T row_max = -INFINITY;
    T sum = static_cast<T>(0.f);

    // max
    for (int j = 0; j < N; j++) {
        row_max = row_max > input[idx * N + j] ? row_max : input[idx * N + j];
    }

    // sum
    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            output[idx * N + j] = static_cast<T>(0.f);
        }
        else {
        output[idx * N + j] = __expf(input[idx * N + j] - row_max);
        }
        sum += output[idx * N + j];
        // sum += exp(input[i] - row_max); is not correct because input[i] is also output[i]
    }

    // softmax
    for (int j = 0; j < N; j++) {
        output[idx * N + j] /= sum;
    }
}

// ============================================================================
// Kernel 3: naive_pv — identical to csrc/naive_attention.cu
// ============================================================================

// QK[M, M] @ V[M, N]
template <typename T>
__global__ void naive_pv(
    const T *P,
    const T *V,
    T *O,
    unsigned int M, unsigned int N
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        T sum = static_cast<T>(0.f);
        for (int m = 0; m < M; m++) {
            sum += P[idx * M + m] * V[(blockDim.x * blockIdx.x + m) * N + j];
        }
        O[idx * N + j] = sum;
    }
}

// ============================================================================
// Host wrapper — uses half, matches csrc/naive_attention.cu launch pattern
// ============================================================================

std::vector<float> naive_attention(
    const std::vector<float>& h_Q_f,
    const std::vector<float>& h_K_f,
    const std::vector<float>& h_V_f,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
    half sm_scale = __float2half(1.0f / sqrtf((float)D));

    const size_t num_elems   = (size_t)B * H * S * D;
    const size_t qkv_bytes   = num_elems * sizeof(half);
    const size_t attn_elems  = (size_t)B * H * S * S;

    // Convert float input to half on host
    std::vector<half> h_Q(num_elems), h_K(num_elems), h_V(num_elems);
    for (size_t i = 0; i < num_elems; i++) {
        h_Q[i] = __float2half(h_Q_f[i]);
        h_K[i] = __float2half(h_K_f[i]);
        h_V[i] = __float2half(h_V_f[i]);
    }

    // Allocate device memory
    half *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    cudaMalloc((void**)&d_Q, qkv_bytes);
    cudaMalloc((void**)&d_K, qkv_bytes);
    cudaMalloc((void**)&d_V, qkv_bytes);
    cudaMalloc((void**)&d_O, qkv_bytes);

    // sm allocation (thrust::device_vector, matches original)
    thrust::device_vector<half> d_sm(attn_elems);
    half* d_sm_ptr = d_sm.data().get();

    // H2D
    cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice);

    // sm = QK^T
    dim3 qk_grid(B * H, 1, 1);
    dim3 qk_block(S, 1, 1);
    naive_gemm<half><<<qk_grid, qk_block, 0, stream>>>(d_Q, d_K, d_sm_ptr, sm_scale, __float2half(0.f), S, S, D);
    cudaStreamSynchronize(stream);

    // softmax
    dim3 sm_grid(B * H, 1, 1);
    dim3 sm_block(S, 1, 1);
    naive_softmax<half><<<sm_grid, sm_block, 0, stream>>>(d_sm_ptr, d_sm_ptr, S);
    cudaStreamSynchronize(stream);

    // O = sm * V
    dim3 o_grid(B * H, 1, 1);
    dim3 o_block(S, 1, 1);
    naive_pv<half><<<o_grid, o_block, 0, stream>>>(d_sm_ptr, d_V, d_O, S, D);
    cudaStreamSynchronize(stream);

    // D2H (half -> float)
    std::vector<half> h_O_half(num_elems);
    cudaMemcpy(h_O_half.data(), d_O, qkv_bytes, cudaMemcpyDeviceToHost);

    std::vector<float> output(num_elems);
    for (size_t i = 0; i < num_elems; i++) {
        output[i] = __half2float(h_O_half[i]);
    }

    // Free (d_sm auto-freed by thrust destructor)
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return output;
}

// ============================================================================
// CPU reference (fp32 for accuracy)
// ============================================================================

std::vector<float> naive_attention_cpu_ref(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int B, int H, int S, int D)
{
    float scale = 1.0f / sqrtf((float)D);
    std::vector<float> O(B * H * S * D, 0.0f);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int qkv_off = b * (H * S * D) + h * (S * D);

            for (int i = 0; i < S; i++) {
                std::vector<float> scores(S);
                float row_max = -INFINITY;

                for (int j = 0; j < S; j++) {
                    if (j > i) {
                        scores[j] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++) {
                            dot += Q[qkv_off + i * D + d] * K[qkv_off + j * D + d];
                        }
                        scores[j] = dot * scale;
                    }
                    if (scores[j] > row_max) row_max = scores[j];
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < S; j++) {
                    scores[j] = std::exp(scores[j] - row_max);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < S; j++) {
                    scores[j] /= sum_exp;
                }

                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < S; j++) {
                        acc += scores[j] * V[qkv_off + j * D + d];
                    }
                    O[qkv_off + i * D + d] = acc;
                }
            }
        }
    }
    return O;
}

// ============================================================================
// Verification — wider tolerance for fp16 (half has ~3 decimal digits precision)
// ============================================================================

static bool close_enough(float out, float gold)
{
    float abs_err = std::abs(out - gold);
    float rel_err = abs_err / std::max(std::abs(gold), 1e-30f);
    return abs_err < 5e-2f || rel_err < 1e-1f;
}

uint32_t verify_result(const std::vector<float>& output,
                       const std::vector<float>& golden,
                       const char* case_name,
                       bool print_samples)
{
    if (output.size() != golden.size()) {
        std::cout << "[Failed] " << case_name << ": size mismatch "
                  << output.size() << " vs " << golden.size() << std::endl;
        return 1;
    }

    uint32_t mismatches  = 0;
    float    max_abs_err = 0.0f;
    float    max_rel_err = 0.0f;
    size_t   first_bad   = output.size();

    for (size_t i = 0; i < output.size(); ++i) {
        float abs_err = std::abs(output[i] - golden[i]);
        float rel_err = abs_err / std::max(std::abs(golden[i]), 1e-30f);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (!close_enough(output[i], golden[i])) {
            if (first_bad == output.size()) first_bad = i;
            ++mismatches;
        }
    }

    if (print_samples) {
        auto dump = [](const char* name, const std::vector<float>& v, size_t n = 16) {
            std::cout << name << ": ";
            for (size_t i = 0; i < std::min(v.size(), n); ++i)
                std::cout << v[i] << " ";
            if (v.size() > n) std::cout << "...";
            std::cout << std::endl;
        };
        dump("Output", output);
        dump("Golden", golden);
    }

    std::cout << "[" << case_name << "] "
              << "max_abs_err=" << max_abs_err
              << " max_rel_err=" << max_rel_err
              << " mismatches=" << mismatches << "/" << output.size();

    if (mismatches == 0) {
        std::cout << " => PASS" << std::endl;
        return 0;
    }
    std::cout << " (first_bad@" << first_bad << ") => FAIL" << std::endl;
    return 1;
}

// ============================================================================
// Test data generation (generate in float, kernel runs in half)
// ============================================================================

static void gen_qkv(std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V,
                    int B, int H, int S, int D, bool debug)
{
    size_t n = (size_t)B * H * S * D;
    Q.resize(n);
    K.resize(n);
    V.resize(n);

    if (debug) {
        for (size_t i = 0; i < n; i++) {
            Q[i] = (float)i * 0.01f - 0.5f;
            K[i] = (float)i * 0.01f;
            V[i] = (float)i * 0.01f + 0.5f;
        }
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n; i++) {
            Q[i] = dist(rng);
            K[i] = dist(rng);
            V[i] = dist(rng);
        }
    }
}

// ============================================================================
// main
// ============================================================================

int32_t main(int32_t argc, char* argv[])
{
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // --- T1: Debug shape [1,1,8,8] ---
    {
        std::vector<float> Q, K, V;
        gen_qkv(Q, K, V, DEBUG_B, DEBUG_H, DEBUG_S, DEBUG_D, true);
        auto golden = naive_attention_cpu_ref(Q, K, V, DEBUG_B, DEBUG_H, DEBUG_S, DEBUG_D);
        auto output = naive_attention(Q, K, V, DEBUG_B, DEBUG_H, DEBUG_S, DEBUG_D, stream);
        if (verify_result(output, golden, "T1 debug [1,1,8,8]", true) != 0) {
            cudaStreamDestroy(stream);
            cudaDeviceReset();
            return 1;
        }
    }

    // --- T2: Perf shape [2,4,256,64] ---
    {
        std::vector<float> Q, K, V;
        gen_qkv(Q, K, V, PERF_B, PERF_H, PERF_S, PERF_D, false);
        auto golden = naive_attention_cpu_ref(Q, K, V, PERF_B, PERF_H, PERF_S, PERF_D);
        auto output = naive_attention(Q, K, V, PERF_B, PERF_H, PERF_S, PERF_D, stream);
        if (verify_result(output, golden, "T2 perf [2,4,256,64]", false) != 0) {
            cudaStreamDestroy(stream);
            cudaDeviceReset();
            return 1;
        }
    }

    std::cout << "[Success] All cases passed." << std::endl;
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    return 0;
}
