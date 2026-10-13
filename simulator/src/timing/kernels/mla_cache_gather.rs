//! Plain MLA cache gather: copy whole cached contexts from the paged MLA cache
//! into a contiguous workspace, dequantizing an FP8 cache as it copies (vLLM
//! `ops.gather_and_maybe_dequant_cache`).
//!
//! Static identity is the entry widths, page size, cache / workspace dtypes
//! and cache format. Runtime is how many requests are gathered and how many
//! rows in total. The call is one launch of `ceil(rows / block_size) + 2 x
//! requests` CTAs, each copying one page, so the time is near linear in rows
//! with a per-request term; the cache interpolates over `(requests, rows)` and
//! past the grid holds the edge's bandwidth. The profiler splits the rows as
//! evenly as possible over the requests.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Requests gathered in one call: a step's prefilling requests, which the
/// chunked-prefill budget keeps to a few at long contexts.
const REQUEST_AXIS: [u32; 4] = [1, 2, 4, 8];
/// Gathered rows: one page and below, then powers of two up to two full
/// 262,144-token contexts (the workspace holds one plus a scheduler step).
const TOKEN_AXIS: [u32; 15] = [
    1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 393216, 524288,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaCacheGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub kv_lora_rank: Dim,
    pub rope_dim: Dim,
    pub block_size: u32,
    #[kv_dtype]
    pub cache_dtype: DType,
    #[compute_dtype]
    pub output_dtype: DType,
    pub cache_format: String,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaCacheGatherKernelInput {
    pub num_requests: u32,
    pub num_tokens: u32,
}

/// The profiler's logical bytes: every row read from the cache and written to
/// the workspace, one int32 block id per page, and the `cu_seq_lens` array.
fn logical_bytes(config: &MlaCacheGatherKernelConfig, input: &MlaCacheGatherKernelInput) -> f64 {
    let requests = u64::from(input.num_requests.max(1));
    let tokens = u64::from(input.num_tokens);
    let (base, extra) = (tokens / requests, tokens % requests);
    let block = u64::from(config.block_size);
    let pages = extra * (base + 1).div_ceil(block) + (requests - extra) * base.div_ceil(block);
    let entry = f64::from(config.kv_lora_rank.get() + config.rope_dim.get());
    let row_bytes =
        entry * f64::from(config.cache_dtype.size_bytes() + config.output_dtype.size_bytes());
    tokens as f64 * row_bytes + 4.0 * pages as f64 + 4.0 * (requests + 1) as f64
}

pub struct MlaCacheGatherSpec;

impl KernelSpec for MlaCacheGatherSpec {
    type Config = MlaCacheGatherKernelConfig;
    type Input = MlaCacheGatherKernelInput;

    const KIND: KernelKind = "mla_cache_gather";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(REQUEST_AXIS.iter().copied()),
            Axis::values(TOKEN_AXIS.iter().copied()),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Rows and requests add work (pages and per-request CTAs) rather than
        // multiply it.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    /// Past the grid the copy holds its bandwidth: it only moves rows.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        OffGrid::Bandwidth(logical_bytes(config, input))
    }

    /// Every request gathers at least one row.
    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|requests, tokens| tokens < requests)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|requests, tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_requests", requests as u32)
                .with("num_tokens", tokens as u32)
                .with("kv_lora_rank", config.kv_lora_rank.get())
                .with("rope_dim", config.rope_dim.get())
                .with("block_size", config.block_size)
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("cache_format", config.cache_format.clone())
        })
    }
}

register_kernel!(MlaCacheGatherKernel, MlaCacheGatherSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MlaCacheGatherKernelConfig {
        MlaCacheGatherKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".into(),
            kv_lora_rank: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 0),
            block_size: 64,
            cache_dtype: DType::Fp8E4m3,
            output_dtype: DType::Bf16,
            cache_format: "plain".into(),
        }
    }

    /// Catches a payload the Python schema rejects (a field renamed, added or
    /// missing) and dtype tags that would let the launcher pick a backend for
    /// the wrong cache.
    #[test]
    fn payloads_carry_the_python_args_and_the_cache_dtype_tag() {
        let cfg = config();
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        let grid = MlaCacheGatherSpec::sweep_grid(&cfg);
        let payloads = MlaCacheGatherSpec::enumerate(&cfg, &grid, "vllm_cuda");
        let fields = serde_json::to_value(payloads[0].fields()).unwrap();
        assert_eq!(
            fields,
            serde_json::json!({
                "backend": "vllm_cuda",
                "num_requests": 1,
                "num_tokens": 1,
                "kv_lora_rank": 512,
                "rope_dim": 0,
                "block_size": 64,
                "cache_dtype": "fp8_e4m3",
                "output_dtype": "bf16",
                "cache_format": "plain",
            })
        );
    }

    /// Catches a grid past the 500-coordinate ceiling or one that would ask
    /// the profiler for a request with no rows.
    #[test]
    fn grid_stays_small_and_masks_empty_requests() {
        let cfg = config();
        let grid = MlaCacheGatherSpec::sweep_grid(&cfg);
        let mask = MlaCacheGatherSpec::infeasible_mask(&cfg, &grid);
        assert_eq!(mask.len(), 4 * 15);
        // Only the one-row column masks: 2, 4 and 8 requests.
        assert_eq!(mask.iter().filter(|&&m| m).count(), 3);
    }

    /// Catches off-grid bandwidth held on bytes counted differently from the
    /// profiler's (`profiling/runners/attention/mla_cache_gather.py`).
    #[test]
    fn logical_bytes_match_the_profiler_count() {
        let cfg = config();
        // 200 rows over 3 requests: 67, 67, 66 rows, two pages each.
        let input = MlaCacheGatherKernelInput {
            num_requests: 3,
            num_tokens: 200,
        };
        assert_eq!(
            logical_bytes(&cfg, &input),
            200.0 * 512.0 * 3.0 + 4.0 * 6.0 + 4.0 * 4.0
        );
        assert_eq!(&*input.coords(), &[3.0, 200.0]);
    }
}
