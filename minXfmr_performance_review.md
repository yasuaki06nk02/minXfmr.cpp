# minXfmr.cpp パフォーマンス & メモリ効率レビュー
**llama.cpp 比較分析 | 2026年7月31日**

---

## 📊 エグゼクティブサマリー

**コード規模**: ~5,100 行（llama.cpp: 100k+ 行）  
**設計哲学**: シンプル性・可読性重視 | CPU-first, CUDA は最適化層  
**対応量子化**: Q4_K, Q5_0, Q8_0  
**プラットフォーム**: Linux, Windows, Android ARM64, Jetson Nano  

### 総合評価
✅ **強み**: シンプルで保守性が高い、メモリ管理が明確  
⚠️ **改善余地**: 量子化カーネルの安定化、Attention最適化、段階的GPU活用  

---

**更新履歴 (2026-08-01)**
- **Phase 1 (安定性強化)**: CUDA 量子化パリティの不一致を修正しました。`src/backend/cuda/cuda_backend.cu` の共有タイルデコードを per-thread 要素デコードに変更し、Q4_K / Q5_0 / Q8_0 の CUDA パリティを確認済みです。関連のテストを追加・実行しました: `tests/test_cuda_quant_kernel_parity.cpp`, `tests/test_cuda_quantized_matmul.cpp`, `tests/test_q4k_block_debug.cpp`, `tests/test_q8_block_debug.cpp`。修正はコミットして `main` ブランチへ push 済みです。
- **CI**: `.github/workflows/cuda-quant-parity.yml` に parity ジョブがあり、自己ホスト GPU ランナー向けにパリティチェックを実行できます。
- **Phase 2**: Tile 化（`src/backend/cpu/cpu_matmul_tiled.cpp`）と NEON 実装雛形（`src/backend/cpu/cpu_backend_neon.cpp`）がリポジトリに存在します。Tile 化は CMake フラグ `MINXFMR_ENABLE_TILED_MATMUL` で有効化できます。`softmax` の SIMD 最適化は未着手です。
- **Phase 3 / Phase 4**: 設計テンプレートはあるものの、n-layer GPU オフロードや FlashAttention、KV キャッシュ量子化の本実装は未完了です。

## 1️⃣ メモリ効率分析

### 1.1 メモリ所有権設計（優秀）

```
評価: ⭐⭐⭐⭐⭐ (llama.cppより明確)
```

**minXfmr.cpp の優位性**:
- **明示的所有権**: すべてのメモリに唯一の所有者を設定
- **メモリリーク防止**: move-only Tensor、raw reference による安全な参照
- **トークン生成中の割り当てなし**: ループ内で new/delete なし（llama.cppは部分的には割り当てあり）

**実装例** (`src/tensor/tensor.h`):
```cpp
Tensor(const Tensor&) = delete;  // コピー禁止 → move-only
Tensor(Tensor&&) noexcept = default;
```

**llama.cpp との比較**:
| 項目 | minXfmr.cpp | llama.cpp |
|------|------------|-----------|
| メモリ所有権 | 明確・1対1 | 複雑（shared_ptr多用） |
| ループ内割り当て | なし | 部分的あり |
| メモリプール | なし | あり（複雑） |

---

### 1.2 KV キャッシュ実装（基本的だが効率的）

```
評価: ⭐⭐⭐⭐ (機能は同等、Ring buffer実装)
```

**実装** (`src/cache/kv_cache.h`):
```cpp
std::vector<float> keys_buf;  // layers * seq_max * dim
std::vector<float> vals_buf;
std::vector<size_t> heads;    // Ring buffer 指針
```

**優位性**:
- Ring buffer で固定メモリ使用（seq_max超過時も安全）
- View ベース（コピーなし）で各レイヤーのキャッシュアクセス

**改善機会**:
- 🔴 **KV キャッシュ量子化未実装**: llama.cpp の `ggml_type_quantized_size()` のような機構がない
  - 現在: seq_max × dim × 4 bytes/layer
  - 理想: seq_max × dim × 1-2 bytes/layer（量子化時）
  
**改善案**:
```cpp
// Future: Quantized KV Cache
struct KVCache {
    DataType cache_type;  // F32 or Q4_K
    std::vector<uint8_t> keys_buf;  // 量子化フォーマット
};
```

---

### 1.3 テンポラリ バッファ管理（優秀）

```
評価: ⭐⭐⭐⭐⭐
```

**Thread-local workspace** (`src/backend/cpu/cpu_backend.cpp`):
```cpp
static thread_local Workspace g_workspace{std::vector<float>(), 0};
float* cpu_workspace(size_t n) {
    // ローカル再利用 → malloc overhead なし
}
```

**利点**:
- スレッド間でのメモリ競合なし
- Attention スコア計算時の動的割り当てなし

**比較**:
- llama.cpp: グローバル work context で管理（複雑）
- minXfmr.cpp: thread-local で単純化（優秀）

---

## 2️⃣ 演算処理の高速化分析

### 2.1 行列乗算（Matmul）

```
評価: ⭐⭐⭐ (基本的だが可改善)
```

#### CPU Backend

**現在実装** (`src/backend/cpu/cpu_backend.cpp` L219-378):
```cpp
// 行分割 (m > n) or 列分割 (m < n)
if (!cfg.split_cols) {
    // m = A->rows を分割してマルチスレッド
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t kk = 0; kk < k; ++kk) {
            float av = arow[kk];
            for (size_t j = 0; j < n; ++j)
                orow[j] += av * brow[j];  // キャッシュ非効率！
        }
    }
}
```

**問題点**:
1. 🔴 **キャッシュ局所性が低い**: k ループが最内ループ
   - Brow の全行をスキャン → L1/L2 ミス
   - llama.cpp は tile 化で改善

2. 🔴 **SIMD 未使用**: float スカラーのみ、AVX/NEON 命令なし

3. ⚠️ **逐次dequantization**: 量子化パス（L135-215）では毎回 Q4_K → F32

#### CUDA Backend

**現在実装** (`src/backend/cuda/cuda_backend.cu`):

**優位性**:
- cuBLAS `cublasGemmEx()` で F32 gemm
- Persistent weight buffer caching（重みを GPU に保持）

**問題点**:
1. 🔴 **量子化カーネルは回帰**:
   ```cpp
   // README より
   MINXFMR_CUDA_QUANT = 1  // デフォルト有効だが
   // 実装上の回帰で通常は OFF
   ```

2. ⚠️ **2つの量子化モード**:
   - Parity mode: ホスト側でdequant → GPU転送（低速）
   - Direct mode: GPU上でdequant → 演算（カーネル回帰）

### 2.2 具体的改善提案

#### 提案1: CPU-side Tile化 Matmul

```cpp
// 改善版: M×N×K の tile化（例: 8×32×K）
void cpu_matmul_tiled(const float* A, const float* B, float* C, 
                      size_t m, size_t n, size_t k) {
    const int tile_m = 8, tile_n = 32;
    for (size_t ii = 0; ii < m; ii += tile_m) {
        for (size_t jj = 0; jj < n; jj += tile_n) {
            // Tile (ii:ii+tile_m, jj:jj+tile_n) を計算
            // → L1 キャッシュに収まりやすい
            for (size_t kk = 0; kk < k; ++kk) {
                // inner loop は tile 内で完結
            }
        }
    }
}
```

**期待効果**: 20-30% 高速化（L1/L2 ヒット率向上）

---

#### 提案2: NEON 最適化（Android ARM64 向け）

```cpp
#ifdef __ARM_NEON
void matmul_f32_neon_row_kernel(...) {
    // 4x float を同時処理
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    
    for (size_t t = 0; t < d; t += 4) {
        float32x4_t b_chunk = vld1q_f32(&B[t]);
        float32x4_t a_val = vld1q_dup_f32(&a_row[kk]);
        acc0 = vmlaq_f32(acc0, a_val, b_chunk);
    }
    // 結果を集約
}
#endif
```

**期待効果**: Jetson Nano や Android ARM64 で 2-3x 高速化

---

#### 提案3: 量子化CUDA カーネルの安定化

**現状**:
```cpp
// cuda_backend.cu L47-60
bool cuda_quant_kernels_enabled() {
    // デフォルトは 1 (有効) だが実装が不完全
}
```

**解決策**:

```cpp
// Step 1: Regression テストケース追加
// test: Q4_K matmul の精度検証（CPU vs CUDA）
bool test_q4k_matmul_parity() {
    Tensor* A = test_data_f32_1x2048();
    Tensor* B = test_data_q4k_2048x4096();
    
    Tensor* out_cpu = ..., *out_cuda = ...;
    cpu_matmul(A, B, out_cpu);
    cuda_matmul(A, B, out_cuda);
    
    // element-wise 比較（eps = 1e-4 程度）
    return allclose(out_cpu, out_cuda, 1e-4);
}

// Step 2: Quantized kernel を 2つ実装して compare
// - direct_dequant_kernel: GPU上でdequant（高速だが不安定）
// - staged_dequant_kernel: ホストでdequant, GPU演算（安定）

// Step 3: 環境変数で制御
MINXFMR_CUDA_QUANT_KERNEL = "staged"  // or "direct"
```

---

### 2.3 Attention 実装

```
評価: ⭐⭐⭐ (基本実装, 最適化余地あり)
```

**現在実装** (`src/transformer/attention.cpp` L39-91):

```cpp
bool attention_apply_with_cache(...) {
    for (size_t i = 0; i < q_rows; ++i) {  // クエリのループ
        // 1. スコア計算: Q[i] · K[j] for all j
        for (size_t j = 0; j < k_rows; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < d; ++t)
                s += q[i*d + t] * k[j*d + t];
            scores[j] = s;
        }
        // 2. Softmax
        softmax_row(scores, k_rows);
        
        // 3. Attention 重み × Value
        for (size_t t = 0; t < d; ++t) {
            double acc = 0.0;
            for (size_t j = 0; j < k_rows; ++j)
                acc += scores[j] * v[j*d + t];
            o[i*d + t] = acc;
        }
    }
}
```

**問題点**:
1. 🔴 **計算量O(seq²×d)**: キャッシュサイズ O(seq×d) に加えて
2. 🔴 **FlashAttention 未実装**: llama.cpp が v2/v3 で採用している高速化機構
3. ⚠️ **ブロック化なし**: マルチヘッド並列化なし

**改善提案**:

#### FlashAttention-like 実装（段階的）

```cpp
// Phase 1: ブロック化QK計算
// K×Q = 256×2048×128 → (256/16)×(2048/128)×(128) のブロックに分割
void attention_flash_qk_blocked(...) {
    const int BLOCK_N = 128;  // K を分割
    for (int block_start = 0; block_start < k_rows; block_start += BLOCK_N) {
        int block_end = min(block_start + BLOCK_N, k_rows);
        // Block [block_start:block_end] との interaction を計算
        // → online softmax で中間結果を集約
    }
}

// Phase 2: Attention forward の融合
// QK + Softmax + AV を単一ループで → メモリ帯域幅削減
```

**期待効果**: 30-50% 高速化 + メモリ帯域幅 30% 削減

---

### 2.4 マルチスレッド戦略

```
評価: ⭐⭐⭐⭐ (実装は良いが可改善)
```

**現在実装** (`src/backend/cpu/cpu_backend.cpp` L114-132):

```cpp
CpuMatmulThreadConfig cfg{};
cfg.nthreads = std::thread::hardware_concurrency();
cfg.split_cols = (m < n);  // m < n なら列分割
```

**優位性**:
- 動的スレッド数調整（MINXFMR_CPU_THREADS で override）
- m/n のバランスに応じた分割戦略

**改善機会**:
1. ⚠️ **NUMA 未対応**: 複数ソケットシステムで非効率
   ```cpp
   // 改善案: NUMA affinity を考慮
   #ifdef MINXFMR_NUMA_AWARE
   numa_bind_thread(thread_id);
   #endif
   ```

2. ⚠️ **ハイパースレッド考慮なし**
   ```cpp
   // 改善案: P-core と E-core を区別
   int p_cores = get_p_core_count();
   cfg.nthreads = p_cores;  // E-core は有効活用できないことが多い
   ```

---

## 3️⃣ GPU オフロード & CUDA バックエンド

```
評価: ⭐⭐⭐ (段階的設計は優秀, 実装は途上)
```

### 3.1 設計（優秀）

**CUDA Backend Design** (`20_CUDA_BACKEND.md`):

```
Phase 1: 基本実装 (現在)
 └─ cuBLAS + persistent weight cache

Phase 2: パフォーマンス改善
 └─ pinned memory + CUDA streams

Phase 3: RTX最適化
 └─ FP16 + Tensor Core

Phase 4: 高度な最適化
 └─ Flash Attention + CUDA Graphs
```

### 3.2 実装状況

✅ **完成**:
- cuBLAS gemm （F32）
- Persistent weight buffer caching
- Quantized weight preloading （parity mode）

⚠️ **途上**:
- 🟡 Quantized CUDA kernels（Q4_K/Q5_0/Q8_0）→ 回帰あり
- 🟡 Pinned memory による H2D/D2H オーバーラップ未実装

❌ **未実装**:
- Flash Attention
- n-layer GPU オフロード（llama.cpp の `-ngl` 相当）

### 3.3 n-layer GPU オフロード提案

**実装ロードマップ**:

```cpp
// Future: Layer-wise GPU offload control
struct TransformerConfig {
    int gpu_layers;  // 最初の N レイヤーを GPU に配置
    int cpu_layers;  // 残りは CPU
};

// 推論時:
for (size_t layer = 0; layer < n_layers; ++layer) {
    if (layer < config.gpu_layers) {
        // GPU で実行（重み常駐）
        cuda_transformer_layer(...);
    } else {
        // CPU で実行（CPU メモリから読込）
        cpu_transformer_layer(...);
    }
}
```

**利点**:
- メモリ制約下での最適化（GPU VRAM < モデルサイズ）
- 段階的な GPU 活用（Jetson Nano など限定 VRAM）

**実装例** (`llama.cpp -ngl` 参考):
```bash
# 32層中 16層を GPU で実行
minxfmr_cli model.gguf --gpu-layers 16
```

**期待効果**: 
- GPU VRAM 512MB → 8 層実行可能（Qwen2.5-1.5B の場合）
- 全レイヤーCPU実行比 → 3-5x 高速化

---

## 4️⃣ 定量化（Quantization）実装

```
評価: ⭐⭐⭐ (機能あり, 安定性に課題)
```

### 4.1 サポート状況

| 形式 | CPU | CUDA | 状態 |
|------|-----|------|------|
| F32  | ✅  | ✅   | 完全対応 |
| Q4_K | ✅  | ⚠️   | 回帰あり |
| Q5_0 | ✅  | ⚠️   | 回帰あり |
| Q8_0 | ✅  | ⚠️   | 回帰あり |

### 4.2 CPU 側の実装（優秀）

**Weight-stationary パターン** (`src/backend/cpu/cpu_backend.cpp` L134-215):

```cpp
// Q4_K 用のテンプレート実装
template <size_t BlockElems, size_t BlockSize, typename DequantFn>
void cpu_matmul_quantized_weight_stationary(...) {
    // ウェイトループで dequant を保持
    float tmp[BlockElems];  // 256 要素の一時バッファ
    for (size_t kk = 0; kk < k; ++kk) {
        const uint8_t* brow = bq + kk * row_bytes;
        for (size_t blk = 0; blk < blocks_per_row; ++blk) {
            dequant_block(brow + blk * BlockSize, tmp);  // dequant
            // tmp を使って演算 → L1/L2 キャッシュ効率向上
        }
    }
}
```

**優位性**:
- ブロック単位での dequant → キャッシュ局所性向上
- マルチスレッド対応（行/列分割）

### 4.3 CUDA 側の課題と改善

**現状** (`cuda_backend.cu` L856-955):

```cpp
if (cuda_quant_kernels_enabled()) {
    if (cuda_quant_parity_mode_enabled()) {
        // Mode A: ホストでdequant, GPUで演算（安定）
        get_or_build_persistent_dequant_f32(B, &dBf);
        matmul_rhs_transposed_kernel<<<...>>>(dA, dBf, dC, ...);
    } else {
        // Mode B: GPU上で直接dequant + 演算（不安定）
        matmul_rhs_quant_kernel<...><<<...>>>(dA, dBq, dC, ...);
    }
}
```

**問題**:
- 🔴 Mode B（direct_dequant）: ブロックキャッシュの競合 → 精度低下
- 🔴 Mode A（staged）: H2D転送が bottleneck（dequant後 3-4x サイズ増加）

**改善案（Hybrid アプローチ）**:

```cpp
// Quantized GEMM を複数の実装オプションで
enum class Q4K_KernelMode {
    STAGED,      // ホストでdequant（最安定）
    ATOMIC_SAFE, // GPU上で原子操作でdequant（安定）
    DIRECT,      // キャッシュ最適化版dequant（最速だが検証必須）
};

// 実装の切り替え
#define ENABLE_QUANTIZED_KERNEL_VARIANTS 1

// テストモード
MINXFMR_CUDA_QUANT_KERNEL_MODE = "atomic_safe"
// 本番モード（完全検証後）
MINXFMR_CUDA_QUANT_KERNEL_MODE = "direct"
```

---

## 5️⃣ Android ARM64 対応

```
評価: ⭐⭐⭐⭐ (設計良好, SIMD実装途上)
```

### 5.1 現状

✅ **完成**:
- GGUF ローダ
- Tokenizer
- CPU 推論
- JNI バインディング（計画）

⚠️ **改善機会**:
- NEON SIMD 最適化（部分的）
- メモリプレッシャー管理（API 計画あり）

### 5.2 NEON 最適化の優先度

**高優先度**:
1. matmul: 256x2048 → F32 演算の 2-3x 高速化
2. softmax: seq² 個の指数計算
3. Q4_K dequant: ブロック単位の高速化

**実装テンプレート** (`src/backend/cpu/cpu_backend.cpp` 新規):

```cpp
// NEON版 Q4_K dequant ブロック
#ifdef __ARM_NEON
void dequant_q4_k_block_neon(const uint8_t* blk, float* dst256) {
    // fp16 → fp32 (NEON: vcvt_f32_f16)
    float16x4_t hd_lo = vld1_f16((float16_t*)(blk + 0));
    float32x4_t d = vcvt_f32_f16(hd_lo);  // bf16 変換
    
    // スケール抽出と乗算（SIMD化）
    uint8x16_t scales = vld1q_u8(blk + 4);
    // 量子化値の 16個を同時処理
    
    for (int i = 0; i < 256; i += 16) {
        // 16個の 4-bit 値を展開
        uint8_t q_byte = ((uint8_t*)blk)[16 + i/2];
        // bit 操作で 4-bit → 8-bit に拡張
        uint8x16_t q_expanded = /* bit operations */;
        // dequant: (q - zero) * scale
    }
}
#endif
```

**期待効果**: ARM64 で 2-4x 高速化

---

## 6️⃣ メモリレイアウト最適化

```
評価: ⭐⭐⭐⭐ (Row-major は良い, トランスポーズ工夫可能)
```

### 6.1 現在のレイアウト

```cpp
// Row-major (llama.cpp, PyTorch と同じ)
Tensor [rows x cols]
index = row * cols + col
```

### 6.2 トランスポーズ戦略（既実装）

**Attention の Wq, Wk, Wv, Wo** (`transformer.cpp` L15-33):

```cpp
static bool g_transpose_square_wq = false;
// 各重みについて個別に行/列トランスポーズを制御

// CLI で指定:
// --transpose-wq         // Wq を転置して使用
// --no-transpose-wq      // 通常のまま
// --transpose-square     // 全square行列を転置
```

**スクリプト** (`scripts/find_best_transpose.py`):
- 16通りの組み合わせをテスト
- 最適なトランスポーズ戦略を自動検出

**改善提案**:

```cpp
// 1. 動的トランスポーズ選択をモデルメタデータから自動決定
// GGUF に architecture hint を保存
// - "llama", "mistral" → square transpose ON
// - "gptneox", "mpt" → OFF

// 2. 重み読込時に物理的にトランスポーズ
// メモリ消費は変わらず、runtime overhead なし
if (should_transpose_wq) {
    Tensor* W_transposed = tensor_transpose_f32(W_original);
    // 以降、常に W_transposed を使用
    tensor_free(W_original);
}
```

---

## 7️⃣ ボトルネック分析 & 最適化ロードマップ

### 7.1 相対的コスト（推定）

**Qwen2.5-1.5B, seq_len=1 トークン生成時**:

```
Attention QK計算      : 40%  (2048×1×128 × 2048 = 256M ops)
Attention AV計算      : 20%  (1×2048×128)
FFN (up+gate)         : 25%  (1×5632)
FFN (down)            : 10%  (5632×1536)
その他（Embedding等）  : 5%
```

### 7.2 最適化優先順位

| 順位 | 項目 | 現状 | 期待改善 | 推定効果 |
|------|------|------|---------|---------|
| 1 | Attention QK最適化 | 基本 | FlashAttention | 40-50% 削減 |
| 2 | CUDA 量子化カーネル | 不安定 | 完全検証 + 最適化 | 20-30% 高速化 |
| 3 | CPU SIMD (NEON/AVX) | 未実装 | 完全実装 | 2-3x 高速化 |
| 4 | n-layer GPU offload | 未実装 | 段階的実装 | 3-5x 高速化 (GPU VRAM制約下) |
| 5 | KV キャッシュ量子化 | 未実装 | Q4 量子化 | メモリ -75% |

---

## 8️⃣ 実装優先度 & マイルストーン案

### Phase 1: 安定性強化（2-3週間）
```
[ ] CUDA 量子化カーネルの回帰テスト & 修正
    - test_quantized_matmul_parity() 追加
    - Regression detection CI
[ ] 量子化フォーマットの統一テスト
    - Q4_K, Q5_0, Q8_0 の cross-verify
```

### Phase 2: CPU 側高速化（3-4週間）
```
[ ] Tile化 Matmul 実装 (ARM NEON 優先)
    - 8×32×K の tile loop
    - Expected: +20-30%
[ ] NEON Q4_K dequant 最適化
    - Parallel block processing
    - Expected: +2-3x (ARM64)
[ ] SIMD softmax （seq length が小さい場合）
```

### Phase 3: GPU 活用（4-5週間）
```
[ ] n-layer GPU offload 実装
    - gpu_layers パラメータ追加
    - 段階的レイヤー配置
[ ] Pinned memory + CUDA streams
    - H2D/D2H overlap
    - Expected: +20-30%
```

### Phase 4: 高度な最適化（5-6週間）
```
[ ] FlashAttention-like ブロック化
    - online softmax
    - Expected: +40-50%
[ ] KV キャッシュ量子化
    - Selective quantization
    - Expected: Memory -75%, Speed -10-20%
```

---

## 9️⃣ llama.cpp との詳細比較

### 比較表

| 領域 | minXfmr.cpp | llama.cpp | 勝者 |
|------|------------|-----------|------|
| **コード行数** | 5.1K | 100K+ | minXfmr (保守性) |
| **メモリ明確性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | minXfmr |
| **CPU Matmul** | 基本的 | Tile化 + SIMD | llama |
| **CUDA gemm** | cuBLAS | cuBLAS + 最適化 | llama（同等） |
| **Attention** | 基本実装 | FlashAttention | llama（2-3x高速） |
| **量子化カーネル** | 不安定 | 安定 + 最適化 | llama |
| **GPU オフロード** | 未実装 | `-ngl` で完全実装 | llama |
| **Android対応** | 計画中 | あり | llama |
| **メモリ効率** | シンプル | 複雑だが効率的 | llama（メモリ効率） |

### 結論

- **minXfmr.cpp 強み**: シンプル性・保守性・明確な設計
- **llama.cpp 強み**: パフォーマンス・機能完成度

**推奨戦略**: 
- minXfmr.cpp を **参照実装・教育用** に
- 本番・高速化は llama.cpp/vLLM で検証後、minXfmr に backport

---

## 🔟 まとめ & アクションアイテム

### ✅ 良い点
1. シンプルで保守性が高い設計
2. メモリ所有権が明確
3. Move-only Tensor で安全
4. Thread-local workspace で効率的
5. 段階的 CUDA 設計が優秀

### ⚠️ 改善すべき点
1. CUDA 量子化カーネルの安定化（高優先度）
2. CPU SIMD（ARM NEON, AVX）未実装
3. FlashAttention 未実装
4. n-layer GPU オフロード未実装
5. KV キャッシュ量子化未実装

### 🎯 次のステップ
```
Week 1-2:  CUDA 量子化カーネル regression test & fix
Week 3-4:  ARM NEON Matmul / dequant 実装
Week 5-8:  n-layer GPU offload + FlashAttention 検証
Week 9+:   KV キャッシュ量子化実装
```

---

**分析日**: 2026年7月31日  
**対象**: minXfmr.cpp main ブランチ  
**比較対象**: llama.cpp（2026年時点の標準実装）
