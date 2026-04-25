/**
 * naive_attention.cu — CUDA implementation of Naive Causal Self-Attention.
 *
 * Formula:  O = softmax(causal_mask(Q @ K^T / sqrt(d))) @ V
 * Input:    Q, K, V, shape [B, H, S, D], fp32
 * Output:   O, shape [B, H, S, D], fp32
 *
 * 3 kernels: naive_gemm (QK^T + scale + mask) -> naive_softmax -> naive_pv (P @ V)
 * Grid: B*H blocks, S threads per block (one thread per row).
 *
 * Mirrored from flash-attn-101/csrc/naive_attention.cu (fp16 template -> fp32 direct).
 * Structure matches ascendc/naive_attention.asc for side-by-side comparison.
 */

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

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
// Kernel 1: naive_gemm — S = scale * (Q @ K^T) + causal mask
// ============================================================================

__global__ void naive_gemm(
    const float* Q,
    const float* K,
    float* S,
    float scale,
    int seq_len,
    int head_dim)
{
    int qkv_offset = blockIdx.x * (seq_len * head_dim);
    int s_offset   = blockIdx.x * (seq_len * seq_len);
    int row = threadIdx.x;

    for (int col = 0; col < seq_len; col++) {
        if (col > row) {
            S[s_offset + row * seq_len + col] = -INFINITY;
        } else {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[qkv_offset + row * head_dim + d]
                     * K[qkv_offset + col * head_dim + d];
            }
            S[s_offset + row * seq_len + col] = scale * dot;
        }
    }
}

// ============================================================================
// Kernel 2: naive_softmax — 3-pass row-wise softmax
// ============================================================================

__global__ void naive_softmax(
    float* input,
    float* output,
    int seq_len)
{
    int offset = blockIdx.x * (seq_len * seq_len);
    int row = threadIdx.x;

    // Pass 1: row max
    float row_max = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        float val = input[offset + row * seq_len + j];
        if (val > row_max) row_max = val;
    }

    // Pass 2: exp(x - max) and sum
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (j > row) {
            output[offset + row * seq_len + j] = 0.0f;
        } else {
            output[offset + row * seq_len + j] = __expf(input[offset + row * seq_len + j] - row_max);
        }
        sum += output[offset + row * seq_len + j];
    }

    // Pass 3: normalize
    for (int j = 0; j < seq_len; j++) {
        output[offset + row * seq_len + j] /= sum;
    }
}

// ============================================================================
// Kernel 3: naive_pv — O = P @ V
// ============================================================================

__global__ void naive_pv(
    const float* P,
    const float* V,
    float* O,
    int seq_len,
    int head_dim)
{
    int p_offset   = blockIdx.x * (seq_len * seq_len);
    int qkv_offset = blockIdx.x * (seq_len * head_dim);
    int row = threadIdx.x;

    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += P[p_offset + row * seq_len + j]
                 * V[qkv_offset + j * head_dim + d];
        }
        O[qkv_offset + row * head_dim + d] = acc;
    }
}

// ============================================================================
// Host wrapper
// ============================================================================

std::vector<float> naive_attention(
    const std::vector<float>& h_Q,
    const std::vector<float>& h_K,
    const std::vector<float>& h_V,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)D);

    const size_t qkv_bytes  = (size_t)B * H * S * D * sizeof(float);
    const size_t attn_bytes = (size_t)B * H * S * S * sizeof(float);
    const size_t out_elems  = (size_t)B * H * S * D;

    // Allocate device memory
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    float *d_attn = nullptr;
    cudaMalloc((void**)&d_Q,    qkv_bytes);
    cudaMalloc((void**)&d_K,    qkv_bytes);
    cudaMalloc((void**)&d_V,    qkv_bytes);
    cudaMalloc((void**)&d_O,    qkv_bytes);
    cudaMalloc((void**)&d_attn, attn_bytes);

    // H2D
    cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice);

    // Grid/Block
    uint32_t grid  = B * H;
    uint32_t block = S;

    // Kernel 1: QK^T + scale + causal mask
    naive_gemm<<<grid, block, 0, stream>>>(d_Q, d_K, d_attn, scale, S, D);
    cudaStreamSynchronize(stream);

    // Kernel 2: softmax (in-place)
    naive_softmax<<<grid, block, 0, stream>>>(d_attn, d_attn, S);
    cudaStreamSynchronize(stream);

    // Kernel 3: P @ V
    naive_pv<<<grid, block, 0, stream>>>(d_attn, d_V, d_O, S, D);
    cudaStreamSynchronize(stream);

    // D2H
    float* h_O_buf = nullptr;
    cudaMallocHost((void**)&h_O_buf, qkv_bytes);
    cudaMemcpy(h_O_buf, d_O, qkv_bytes, cudaMemcpyDeviceToHost);

    std::vector<float> output(h_O_buf, h_O_buf + out_elems);

    // Free
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_attn);
    cudaFreeHost(h_O_buf);

    return output;
}

// ============================================================================
// CPU reference
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
// Verification — tolerance-based (matmul + exp accumulate float error)
// ============================================================================

static bool close_enough(float out, float gold)
{
    float abs_err = std::abs(out - gold);
    float rel_err = abs_err / std::max(std::abs(gold), 1e-30f);
    return abs_err < 1e-4f || rel_err < 1e-4f;
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
// Test data generation
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
