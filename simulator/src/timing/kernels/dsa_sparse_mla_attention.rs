//! DSA sparse MLA attention kernel.
//!
//! The cache uses physical `(num_queries, num_cache_tokens)` coordinates. The
//! static `valid_counts_pattern` deterministically derives Python's canonical
//! flattened valid-slot-count RLE during enumeration; RLE text is never a cache
//! coordinate. The first R.4 gate found pattern-specific interpolation defects:
//! uniform adds S6/S11/S91, causal pins its domain boundary and scattered-cache
//! misses, and speculative removes unsupported odd-Q grid holes while adding
//! Q134/Q266 and S3/S23/S45/S182. Cache math and the public query contract are
//! unchanged.
//!
//! A speculative verify step submits one group of `next_n` query rows per
//! request, so the group width is a physical property of the pattern rather than
//! a fixed two. `ValidCountsPattern::SpeculativeGroups` carries it, and derives
//! its axes from the group-of-two anchors by request count.
//!
//! GLM-5.3-Flash's pooled indexer (kpool) selects `index_topk` pools plus the
//! trailing partial pool, so a row's active count is
//! `min(n, index_topk + n mod index_kpool) <= index_topk + index_kpool - 1`
//! (2051), inside a page table `round_up(2051, 128) = 2176` wide
//! (`selected_k`). `ValidCountsPattern::PooledUniformFull` carries the pooling
//! so decode derives `u:min(ctx, index_topk + index_kpool - 1)` instead of
//! clipping at the page-table width, and sweeps its own smaller grid.
//!
//! On H200 the same model runs vLLM's `FLASH_ATTN_MLA_SPARSE`
//! (`vllm_fa3_sparse_bf16`): one FA3 call is `prepare_varlen_num_blocks`,
//! `FlashAttnFwdSm90` and, at 992 query rows or fewer, `FlashAttnFwdCombine`
//! (measured on H200 at 1/256/2176 valid slots: 3 launches through Q 992, 2 from
//! Q 993). Its pooled grid carries 992 and 993 as adjacent query points, so the
//! linear cache never interpolates across that drop.
//!
//! With nvcc on PATH vLLM picks `FLASHINFER_MLA_SPARSE_SM90` instead
//! (`vllm_flashinfer_sparse_sm90_bf16`): FlashInfer's planned
//! `BatchMLAPagedAttentionWrapper` run, one `BatchMLAPageAttentionHopperKernel`
//! launch over the same rows. Its planner (`MLAPlan`, scheduler.cuh) gives each
//! of the H200's 132 SMs one cluster and caps a work item at
//! `f(ceil(total planned keys / 132))`, rounded up to 256 keys. Once a call
//! has more than 132 rows that cap is at least a whole row, so rows of equal
//! length are placed whole, `ceil(Q / 132)` per cluster: the time is a staircase
//! in Q (measured on H200 at 4096 cache tokens: 63.5 us at Q 132, 119.1 us at
//! 133, 121.5 us at 264, 176.8 us at 265). The grid therefore samples Q at full
//! waves above 132 and `cache_coords` rounds a query up to its wave, so the
//! linear cache interpolates in waves and never across a step.
//!
//! With `--kv-cache-dtype fp8` the same backend runs over e4m3 latent rows
//! (`vllm_flashinfer_sparse_sm90_fp8`, layout `token_major_mqa_fp8_latent`):
//! FlashInfer's Hopper kernel loads the FP8 tiles and repacks them to BF16 in
//! shared memory, measured 2.8-3.3x the BF16 cache's time per call on H200.
//! It shares the BF16 backend's planner, grid and wave projection.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const UNIFORM_QUERY_AXIS: [u32; 27] = [
    1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 131, 132, 133, 255, 256, 257, 263, 264, 265, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
];
const UNIFORM_CACHE_AXIS: [u32; 30] = [
    1, 2, 4, 6, 8, 11, 16, 32, 63, 64, 65, 91, 127, 128, 129, 256, 512, 1024, 2047, 2048, 2049,
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
const CAUSAL_QUERY_AXIS: [u32; 37] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 127, 128, 129, 130, 131, 132, 133, 200, 254, 255,
    256, 257, 258, 263, 264, 265, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const CAUSAL_CACHE_AXIS: [u32; 41] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 63, 64, 65, 90, 127, 128, 129, 130, 182, 200, 255, 256,
    260, 511, 512, 513, 724, 1024, 2047, 2048, 2049, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576,
];
const SPECULATIVE_QUERY_AXIS: [u32; 20] = [
    2, 4, 8, 16, 32, 64, 128, 132, 134, 256, 264, 266, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536,
];
const SPECULATIVE_CACHE_AXIS: [u32; 31] = [
    1, 2, 3, 4, 8, 16, 23, 32, 45, 63, 64, 65, 127, 128, 129, 182, 256, 512, 1024, 2047, 2048,
    2049, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
/// Pooled decode queries stop at the TRTLLM 32,768-query launch guard, so the
/// grid carries no cell the backend must mask. 96 splits the 64..127 decode
/// batch gap, where the saturated time bends (B200 b3 fidelity).
const POOLED_QUERY_AXIS: [u32; 21] = [
    1, 2, 4, 8, 16, 32, 64, 96, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 4096, 8192, 16384,
    32768,
];
/// Pooled context points below and above the saturating count. The
/// pooling-derived `index_topk` and cap points are added by
/// `pooled_cache_axis`; past the cap only the gather footprint grows.
/// 768 splits 512..1024, where small-batch time dips then climbs.
const POOLED_CACHE_AXIS: [u32; 21] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 1048576,
];

/// The H200 FA3 backend (vLLM `FLASH_ATTN_MLA_SPARSE`).
pub const FA3_SPARSE_BACKEND: &str = "vllm_fa3_sparse_bf16";
/// Largest FA3 call that still splits the keys and launches
/// `FlashAttnFwdCombine` (FA3's dynamic split runs only at batch <= 992).
pub const FA3_COMBINE_MAX_QUERIES: u32 = 992;
/// FA3 pooled query axis. 992 / 993 bracket the Combine drop as adjacent
/// integer points; the rest is dense at small decode batches and
/// log-spaced above.
const FA3_POOLED_QUERY_AXIS: [u32; 23] = [
    1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 992, 993, 1024, 1536, 2048, 4096,
    8192, 16384, 32768,
];
/// FA3 pooled context points; `pooled_cache_axis` adds `index_topk` and the
/// cap. Sparser than `POOLED_CACHE_AXIS` past the cap, where only the gather
/// footprint grows, to stay within 500 coordinates.
const FA3_POOLED_CACHE_AXIS: [u32; 18] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 4096, 16384, 65536, 262144, 1048576,
];

/// The H200 FlashInfer backend (vLLM `FLASHINFER_MLA_SPARSE_SM90`).
pub const FLASHINFER_SM90_SPARSE_BACKEND: &str = "vllm_flashinfer_sparse_sm90_bf16";
/// The same FlashInfer call over an FP8 (e4m3) latent cache.
pub const FLASHINFER_SM90_SPARSE_FP8_BACKEND: &str = "vllm_flashinfer_sparse_sm90_fp8";

/// Whether `backend` is one of the FlashInfer SM90 backends, which share the
/// planner's wave-aligned grid and query projection.
fn is_flashinfer_sm90(backend: &str) -> bool {
    backend == FLASHINFER_SM90_SPARSE_BACKEND || backend == FLASHINFER_SM90_SPARSE_FP8_BACKEND
}

/// The config prices exactly one FlashInfer SM90 backend.
fn single_flashinfer_sm90(config: &DsaSparseMlaAttentionKernelConfig) -> bool {
    matches!(config.backends.as_slice(), [backend] if is_flashinfer_sm90(backend))
}
/// FlashInfer SM90 pooled query axis: dense below one wave, where the planner
/// splits rows into rounded key chunks and the time is not monotone in Q
/// (128 rows take longer than 132 at 2051 keys), then whole waves of 132 rows
/// (`flashinfer_sm90_wave_rows`), log-spaced past 8 waves. It stops at 124
/// waves: `MLAPlan` holds at most 16384 work items, at least one per row, and
/// overruns its arrays past that (production plans at most 8192 rows).
const FLASHINFER_SM90_POOLED_QUERY_AXIS: [u32; 25] = [
    1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 96, 112, 128, 132, 264, 396, 528, 792, 1056, 1584, 2112,
    4224, 8316, 12408, 16368,
];
/// FlashInfer SM90 pooled context points; the grid adds the cap. The kernel
/// walks keys in 64-token tiles and a partial last tile costs more than a
/// full one: on H200 at 264 rows, 256 keys take 25.6 us, 258..288 keys a
/// plateau near 36.7 us, then the time falls linearly to 29.1 us at 320. A
/// real row's remainder is arbitrary, so below the cap the points sit at
/// 64k + 40, where the time is near its mean over remainders (35.6 vs 35.0
/// us at k = 4; 77.1 vs 76.3 us at k = 16), not on the fast full-tile teeth.
/// Past the cap only the gather footprint grows.
const FLASHINFER_SM90_POOLED_CACHE_AXIS: [u32; 18] = [
    1, 2, 4, 8, 16, 40, 104, 168, 296, 552, 808, 1064, 1576, 2024, 4096, 16384, 65536, 1048576,
];

/// SMs of the GPUs whose FlashInfer SM90 planner rounds is known: `MLAPlan`
/// runs one cluster per SM below 64 packed query rows per request, which a
/// one-query row of at most 64 heads always is.
pub fn flashinfer_sm90_clusters(gpu_name: &str) -> Option<u32> {
    match gpu_name {
        "NVIDIA H200" => Some(132),
        _ => None,
    }
}

/// The query rows the FlashInfer SM90 planner costs a call of `num_queries`
/// equal rows at: past one wave, whole waves of `clusters` rows.
fn flashinfer_sm90_wave_rows(num_queries: u32, clusters: u32) -> u32 {
    if num_queries <= clusters {
        num_queries
    } else {
        num_queries.div_ceil(clusters) * clusters
    }
}

/// Which valid-slot-count shape a config sweeps.
///
/// One choice, not two: the shape decides both the measured grid and the
/// flattened count vector that grid enumerates.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidCountsPattern {
    /// Every query row sees the whole cache.
    UniformFull,
    /// One prefill request's rows walk a causal ramp over its own context.
    CausalTail,
    /// A speculative verify step: `group_size` rows per request, each seeing one
    /// fewer context token than the next.
    SpeculativeGroups { group_size: u32 },
    /// Pooled-indexer decode: every row sees its whole context, but the indexer
    /// keeps at most `index_topk` pools of `index_kpool` tokens plus the
    /// partial tail pool, so the count saturates at
    /// `index_topk + index_kpool - 1`, below the `selected_k` page-table width.
    PooledUniformFull { index_topk: u32, index_kpool: u32 },
}

impl ValidCountsPattern {
    /// Choose the decode shape for a step submitting `verify_width` query rows
    /// per request, or `None` if that is not a width.
    ///
    /// One row is ordinary decode and every row sees the whole cache. Two or
    /// more rows are a verify group walking a descending context ramp. The two
    /// sweep different measured grids — uniform is dense in query count, while
    /// speculative anchors on request count — so this picks a grid, not just an
    /// encoding.
    pub fn for_decode(verify_width: u32) -> Option<Self> {
        match verify_width {
            0 => None,
            1 => Some(Self::UniformFull),
            group_size => Some(Self::SpeculativeGroups { group_size }),
        }
    }

    /// Largest active count a row can reach under this pattern.
    ///
    /// Unpooled patterns clip at the `selected_k` page-table width. The pooled
    /// pattern keeps `index_topk` whole pools plus up to `index_kpool - 1`
    /// trailing tokens (vLLM `_expand_pools_and_append_tail`), which must fit
    /// the page table.
    fn valid_count_cap(self, selected_k: u32) -> u32 {
        match self {
            Self::PooledUniformFull {
                index_topk,
                index_kpool,
            } => {
                assert!(
                    index_topk > 0 && index_kpool > 0,
                    "pooled index_topk and index_kpool must be positive"
                );
                let cap = index_topk
                    .checked_add(index_kpool - 1)
                    .expect("pooled valid-count cap overflows u32");
                assert!(
                    cap <= selected_k,
                    "pooled valid-count cap {cap} exceeds selected_k {selected_k}"
                );
                cap
            }
            Self::UniformFull | Self::CausalTail | Self::SpeculativeGroups { .. } => selected_k,
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub selected_k: u32,
    pub latent_dim: Dim,
    pub rope_dim: Dim,
    pub value_dim: Dim,
    pub softmax_scale_denominator: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub index_dtype: String,
    pub output_dtype: DType,
    pub valid_counts_pattern: ValidCountsPattern,
    pub index_distribution: String,
    pub cache_layout: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelInput {
    pub num_queries: u32,
    pub num_cache_tokens: u32,
}

pub struct DsaSparseMlaAttentionSpec;

impl KernelSpec for DsaSparseMlaAttentionSpec {
    type Config = DsaSparseMlaAttentionKernelConfig;
    type Input = DsaSparseMlaAttentionKernelInput;

    const KIND: KernelKind = "dsa_sparse_mla_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if config.backends.contains(&FA3_SPARSE_BACKEND) {
            return fa3_sweep_grid(config);
        }
        if config.backends.iter().any(|b| is_flashinfer_sm90(b)) {
            return flashinfer_sm90_sweep_grid(config);
        }
        let (query_axis, cache_axis): (Vec<u32>, Vec<u32>) = match config.valid_counts_pattern {
            ValidCountsPattern::UniformFull => {
                (UNIFORM_QUERY_AXIS.to_vec(), UNIFORM_CACHE_AXIS.to_vec())
            }
            ValidCountsPattern::CausalTail => {
                (CAUSAL_QUERY_AXIS.to_vec(), CAUSAL_CACHE_AXIS.to_vec())
            }
            ValidCountsPattern::SpeculativeGroups { group_size } => (
                speculative_query_axis(group_size),
                speculative_cache_axis(group_size, config.selected_k),
            ),
            ValidCountsPattern::PooledUniformFull { index_topk, .. } => (
                POOLED_QUERY_AXIS.to_vec(),
                pooled_cache_axis(
                    index_topk,
                    config
                        .valid_counts_pattern
                        .valid_count_cap(config.selected_k),
                ),
            ),
        };
        SweepGrid::new(vec![
            Axis::values(query_axis.iter().copied()),
            Axis::values(cache_axis.iter().copied()),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Sparse attention reads `selected_k` tokens per query, and `selected_k`
        // is a Config field, not an axis. Work is therefore ~linear in
        // `num_queries`; `num_cache_tokens` sets the gather footprint the
        // indices point into, not how many of them are read.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        if single_flashinfer_sm90(config) {
            let clusters = flashinfer_sm90_clusters(&config.gpu_name)
                .expect("validate_config admits only GPUs with a known SM count");
            return Coords::new([
                f64::from(flashinfer_sm90_wave_rows(input.num_queries, clusters)),
                f64::from(input.num_cache_tokens),
            ]);
        }
        input.coords()
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        if let Some(backend) = config.backends.iter().find(|b| is_flashinfer_sm90(b)) {
            anyhow::ensure!(
                flashinfer_sm90_clusters(&config.gpu_name).is_some(),
                "{backend} rounds queries to the planner's waves, \
                 whose SM count is known only for NVIDIA H200, got {}",
                config.gpu_name
            );
        }
        Ok(())
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // TRTLLM-gen rejects the 65,536-query launch on B200. Production caps
        // one iteration far below that, so 32,768 remains a generous measured
        // extrapolation guard without requiring an unsupported launch.
        let trtllm_query_too_large = |num_queries: f64| {
            config.backends.contains(&"flashinfer_trtllm_fp8") && num_queries > 32_768.0
        };
        match config.valid_counts_pattern {
            ValidCountsPattern::UniformFull | ValidCountsPattern::PooledUniformFull { .. } => {
                grid.expand_2d(|num_queries, _| trtllm_query_too_large(num_queries))
            }
            ValidCountsPattern::CausalTail => grid.expand_2d(|num_queries, num_cache_tokens| {
                num_queries > num_cache_tokens || trtllm_query_too_large(num_queries)
            }),
            ValidCountsPattern::SpeculativeGroups { group_size } => {
                grid.expand_2d(|num_queries, _| {
                    num_queries as u32 % group_size != 0 || trtllm_query_too_large(num_queries)
                })
            }
        }
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(
            config.softmax_scale_denominator > 0,
            "softmax_scale_denominator must be positive"
        );
        grid.expand_2d(|num_queries, num_cache_tokens| {
            let num_queries = num_queries as u32;
            let num_cache_tokens = num_cache_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries)
                .with("num_cache_tokens", num_cache_tokens)
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("selected_k", config.selected_k)
                .with("latent_dim", config.latent_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("value_dim", config.value_dim.get())
                .with(
                    "softmax_scale",
                    1.0 / f64::from(config.softmax_scale_denominator),
                )
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with(
                    "valid_counts",
                    canonical_valid_counts(
                        config.valid_counts_pattern,
                        num_queries,
                        num_cache_tokens,
                        config
                            .valid_counts_pattern
                            .valid_count_cap(config.selected_k),
                    ),
                )
                .with("index_distribution", config.index_distribution.clone())
                .with("cache_layout", config.cache_layout.clone())
        })
    }
}

/// Derive the exact canonical Python valid-count encoding from physical axes.
///
/// `k` is the pattern's valid-count cap (`ValidCountsPattern::valid_count_cap`):
/// `selected_k` for unpooled patterns, `index_topk + index_kpool - 1` pooled.
fn canonical_valid_counts(pattern: ValidCountsPattern, q: u32, s: u32, k: u32) -> String {
    match pattern {
        ValidCountsPattern::UniformFull => format!("u:{}x{q}", s.min(k)),
        ValidCountsPattern::PooledUniformFull {
            index_topk,
            index_kpool,
        } => canonical_pooled_counts(q, s, index_topk, index_kpool),
        ValidCountsPattern::CausalTail => {
            if q > s {
                return masked_placeholder(q);
            }
            let first = s - q + 1;
            let last = s;
            if q == 1 || first >= k {
                format!("u:{}x{q}", first.min(k))
            } else if last <= k {
                format!("r:{first}..{last}")
            } else {
                debug_assert!(first < k && k < last);
                format!("c:{first}..{last}@{k}")
            }
        }
        ValidCountsPattern::SpeculativeGroups { group_size } => {
            if q % group_size != 0 {
                return masked_placeholder(q);
            }
            canonical_speculative_counts(q, s, k, group_size)
        }
    }
}

/// Pooled decode rows past `index_topk` keep `index_topk` pools plus the
/// `n mod index_kpool` tail tokens of their own context `n`. A decode batch's
/// contexts are unrelated, so its rows spread over every tail phase: row `i`
/// sees `min(s, index_topk + i mod index_kpool)`. The phases are not
/// interchangeable on B200 (job 1141, 32 queries): 2048 / 2049 / 2050 / 2051
/// active rows take 16.8 / 26.3 / 21.9 / 18.8 us and the phase-mixed batch
/// 21.9 us, so a single saturated count would misstate the batch by up to 30%.
///
/// The encoding mirrors Python `_encode_valid_counts`: uniform, then a +1
/// ramp, then the shortest repeating group.
fn canonical_pooled_counts(q: u32, s: u32, index_topk: u32, index_kpool: u32) -> String {
    if s <= index_topk {
        return format!("u:{s}x{q}");
    }
    let counts: Vec<u32> = (0..q)
        .map(|row| s.min(index_topk + row % index_kpool))
        .collect();
    if counts.iter().all(|&count| count == counts[0]) {
        return format!("u:{}x{q}", counts[0]);
    }
    if counts
        .iter()
        .enumerate()
        .all(|(row, &count)| count == counts[0] + row as u32)
    {
        return format!("r:{}..{}", counts[0], counts[counts.len() - 1]);
    }
    let len = counts.len();
    let period = (1..=len)
        .find(|&candidate| {
            len % candidate == 0
                && counts
                    .iter()
                    .enumerate()
                    .all(|(row, &count)| count == counts[row % candidate])
        })
        .expect("the whole vector is always a period");
    let group = counts[..period]
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("g:({group})x{}", len / period)
}

/// Scale the frozen group-of-two query anchors by request count.
///
/// Each grid point is `requests * group_size` query rows, so holding the request
/// count fixed keeps the measured batch sizes — and the odd-Q holes the R.4 gate
/// removed — aligned across group sizes.
fn speculative_query_axis(group_size: u32) -> Vec<u32> {
    assert!(group_size > 0, "speculative group_size must be positive");
    if group_size == 2 {
        return SPECULATIVE_QUERY_AXIS.to_vec();
    }
    SPECULATIVE_QUERY_AXIS
        .iter()
        .map(|num_queries| {
            (num_queries / 2)
                .checked_mul(group_size)
                .expect("speculative query axis overflows u32")
        })
        .collect()
}

/// Add the group-sensitive transition points a wider verify group crosses.
///
/// The RLE form changes shape where the group first fits in the context
/// (`num_cache_tokens` near `group_size`) and where it straddles the `selected_k`
/// clip (`selected_k ..= selected_k + group_size - 1`). Without samples at
/// those boundaries the cache interpolates across a discontinuity it never
/// measured. The frozen group-of-two axis is returned untouched.
fn speculative_cache_axis(group_size: u32, selected_k: u32) -> Vec<u32> {
    if group_size == 2 {
        return SPECULATIVE_CACHE_AXIS.to_vec();
    }
    let saturated_context = selected_k
        .checked_add(group_size - 1)
        .expect("speculative saturation boundary overflows u32");
    let mut axis = SPECULATIVE_CACHE_AXIS.to_vec();
    axis.extend(
        [
            group_size - 1,
            group_size,
            group_size + 1,
            selected_k.saturating_sub(group_size - 1),
            selected_k,
            selected_k + 1,
            saturated_context.saturating_sub(1),
            saturated_context,
            saturated_context
                .checked_add(1)
                .expect("speculative saturation guard overflows u32"),
        ]
        .into_iter()
        .filter(|value| *value > 0),
    );
    axis.sort_unstable();
    axis.dedup();
    axis
}

/// Mirror Python's canonical encoder for `q / group_size` repeats of one
/// speculative verify group.
///
/// Row `i` of a group sees `num_cache_tokens - (group_size - 1 - i)` context
/// tokens, clipped to `selected_k` above and to zero below, so the group is
/// non-decreasing and one group determines the whole vector. Building just that
/// group keeps enumeration `O(group_size)` instead of `O(num_queries)` at the
/// largest request anchor, and is exact:
///
/// * `u:` needs every row equal, which is a property of the group alone.
/// * `r:` and `c:` compare against a ramp seeded at row 0. With more than one
///   repeat, row `group_size` is back at the group's first value while the ramp
///   has advanced, and the ramp can only have stalled by reaching `selected_k` —
///   which would have made the group uniform. So both forms require exactly one
///   repeat.
/// * Python's `g:` uses the minimal period. A non-decreasing sequence whose
///   period divides its length is constant, so a non-uniform group's minimal
///   period is the whole group, and Fine and Wilf's theorem rules out a shorter
///   period appearing only once the group is repeated.
fn canonical_speculative_counts(q: u32, s: u32, k: u32, group_size: u32) -> String {
    let group: Vec<u32> = (0..group_size)
        .map(|row| s.saturating_sub(group_size - 1 - row).min(k))
        .collect();
    let first = group[0];
    let last = group[group.len() - 1];
    if first == last {
        return format!("u:{first}x{q}");
    }

    if q == group_size {
        let is_ramp = |cap: u32| {
            group
                .iter()
                .enumerate()
                .all(|(row, &count)| count == (first + row as u32).min(cap))
        };
        if is_ramp(u32::MAX) {
            return format!("r:{first}..{last}");
        }
        let unclipped_last = first + group_size - 1;
        if first < k && k < unclipped_last && unclipped_last <= s && is_ramp(k) {
            return format!("c:{first}..{unclipped_last}@{k}");
        }
    }

    let counts = group
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("g:({counts})x{}", q / group_size)
}

/// The FA3 grid. It differs from the other backends' (the Combine breakpoint),
/// so FA3 sweeps alone, and only the pooled pattern its H200 worklet uses.
fn fa3_sweep_grid(config: &DsaSparseMlaAttentionKernelConfig) -> SweepGrid {
    assert!(
        config.backends == [FA3_SPARSE_BACKEND],
        "{FA3_SPARSE_BACKEND} has its own grid and cannot share a config with {:?}",
        config.backends
    );
    let ValidCountsPattern::PooledUniformFull { index_topk, .. } = config.valid_counts_pattern
    else {
        panic!(
            "{FA3_SPARSE_BACKEND} sweeps only PooledUniformFull, got {:?}",
            config.valid_counts_pattern
        );
    };
    let mut cache_axis = FA3_POOLED_CACHE_AXIS.to_vec();
    cache_axis.extend([
        index_topk,
        config
            .valid_counts_pattern
            .valid_count_cap(config.selected_k),
    ]);
    cache_axis.sort_unstable();
    cache_axis.dedup();
    SweepGrid::new(vec![
        Axis::values(FA3_POOLED_QUERY_AXIS.iter().copied()),
        Axis::values(cache_axis),
    ])
}

/// The FlashInfer SM90 grid: its own wave-aligned query axis, so it sweeps
/// alone, and only the pooled pattern its H200 worklet uses.
fn flashinfer_sm90_sweep_grid(config: &DsaSparseMlaAttentionKernelConfig) -> SweepGrid {
    assert!(
        single_flashinfer_sm90(config),
        "the FlashInfer SM90 backends have their own grid and price one config each, got {:?}",
        config.backends
    );
    let ValidCountsPattern::PooledUniformFull { .. } = config.valid_counts_pattern else {
        panic!(
            "{:?} sweeps only PooledUniformFull, got {:?}",
            config.backends, config.valid_counts_pattern
        );
    };
    // No `index_topk` point: 2048 keys are whole tiles, faster than the
    // partial-tile rows on either side of it.
    let mut cache_axis = FLASHINFER_SM90_POOLED_CACHE_AXIS.to_vec();
    cache_axis.push(
        config
            .valid_counts_pattern
            .valid_count_cap(config.selected_k),
    );
    cache_axis.sort_unstable();
    cache_axis.dedup();
    SweepGrid::new(vec![
        Axis::values(FLASHINFER_SM90_POOLED_QUERY_AXIS.iter().copied()),
        Axis::values(cache_axis),
    ])
}

/// Pooled context axis: the fixed points plus `index_topk` (last unsaturated
/// power-of-two anchor) and the saturating cap, so the linear cache never
/// interpolates across the point where the count stops growing.
fn pooled_cache_axis(index_topk: u32, cap: u32) -> Vec<u32> {
    let mut axis = POOLED_CACHE_AXIS.to_vec();
    axis.extend([index_topk, cap]);
    axis.sort_unstable();
    axis.dedup();
    axis
}

fn masked_placeholder(q: u32) -> String {
    format!("u:0x{q}")
}

register_kernel!(DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionSpec);

#[cfg(test)]
mod tests {
    use super::ValidCountsPattern;
    use super::{
        canonical_valid_counts, speculative_cache_axis, speculative_query_axis,
        DsaSparseMlaAttentionKernelConfig, DsaSparseMlaAttentionKernelInput,
        DsaSparseMlaAttentionSpec, CAUSAL_CACHE_AXIS, CAUSAL_QUERY_AXIS, FA3_COMBINE_MAX_QUERIES,
        FA3_SPARSE_BACKEND, FLASHINFER_SM90_SPARSE_BACKEND, FLASHINFER_SM90_SPARSE_FP8_BACKEND,
        SPECULATIVE_CACHE_AXIS, SPECULATIVE_QUERY_AXIS, UNIFORM_CACHE_AXIS, UNIFORM_QUERY_AXIS,
    };
    use crate::timing::bridge::{ArgsPayload, DType};
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;
    use std::collections::BTreeSet;

    const TORCH_BACKEND: &str = "torch";
    const FLASHMLA_BACKEND: &str = "vllm_flashmla_bf16";
    fn config(pattern: ValidCountsPattern) -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![TORCH_BACKEND, FLASHMLA_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: Dim::param("num_attention_heads", 64),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2048,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 64),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            q_dtype: DType::Bf16,
            cache_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            valid_counts_pattern: pattern,
            index_distribution: "recent_contiguous".to_string(),
            cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
        }
    }

    #[test]
    fn config_kind_and_dtype_identity_match_the_python_handoff() {
        let cfg = config(ValidCountsPattern::UniformFull);

        assert_eq!(DsaSparseMlaAttentionSpec::KIND, "dsa_sparse_mla_attention");
        assert_eq!(
            DsaSparseMlaAttentionSpec::profile_kind(),
            "dsa_sparse_mla_attention"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, FLASHMLA_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.selected_k, 2048);
        assert_eq!(cfg.latent_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.value_dim, 512);
        assert_eq!(cfg.softmax_scale_denominator, 16);
        assert_eq!(cfg.q_dtype, DType::Bf16);
        assert_eq!(cfg.cache_dtype, DType::Bf16);
        assert_eq!(cfg.output_dtype, DType::Bf16);
        assert_eq!(cfg.index_dtype, "int32");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn describe_config_preserves_rich_dimensions_and_static_modes() {
        assert_eq!(
            config(ValidCountsPattern::CausalTail).describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, FLASHMLA_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_heads": rich_dim("num_attention_heads", 64),
                "num_kv_heads": rich_dim("num_kv_heads", 1),
                "selected_k": 2048,
                "latent_dim": rich_dim("kv_lora_rank", 512),
                "rope_dim": rich_dim("qk_rope_head_dim", 64),
                "value_dim": rich_dim("kv_lora_rank", 512),
                "softmax_scale_denominator": 16,
                "q_dtype": "bf16",
                "cache_dtype": "bf16",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts_pattern": ValidCountsPattern::CausalTail,
                "index_distribution": "recent_contiguous",
                "cache_layout": "token_major_mqa_bf16_latent_rope",
            })
        );
    }

    #[test]
    fn input_coordinates_deserialization_and_slot_input_are_physical() {
        let input: DsaSparseMlaAttentionKernelInput =
            serde_json::from_str(r#"{"num_queries":128,"num_cache_tokens":2049}"#).unwrap();

        assert_eq!(&*input.coords(), &[128.0, 2049.0]);
        assert_eq!(
            DsaSparseMlaAttentionKernelInput::coord_field_names(),
            &["num_queries", "num_cache_tokens"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_queries":128,"num_cache_tokens":2049})
        );
    }

    #[test]
    fn grids_have_the_exact_pattern_specific_axes_and_boundaries() {
        assert_grid(
            ValidCountsPattern::UniformFull,
            &UNIFORM_QUERY_AXIS,
            &UNIFORM_CACHE_AXIS,
            810,
        );
        assert_grid(
            ValidCountsPattern::CausalTail,
            &CAUSAL_QUERY_AXIS,
            &CAUSAL_CACHE_AXIS,
            1_517,
        );
        assert_grid(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &SPECULATIVE_QUERY_AXIS,
            &SPECULATIVE_CACHE_AXIS,
            620,
        );

        // Every pattern's cache axis now reaches the arch's full 1,048,576-token
        // timing domain, and every query axis reaches 65,536.
        for pattern in [
            ValidCountsPattern::UniformFull,
            ValidCountsPattern::CausalTail,
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
        ] {
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
            assert_eq!(grid.axes()[0].last(), Some(&65536.0));
            assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        }

        let uniform =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::UniformFull));
        assert_contiguous(&uniform.axes()[0], &[127, 128, 129]);
        assert_contiguous(&uniform.axes()[0], &[131, 132, 133]);
        assert_contiguous(&uniform.axes()[0], &[255, 256, 257, 263, 264, 265]);
        assert_contiguous(&uniform.axes()[1], &[2, 4, 6, 8, 11, 16]);
        assert_contiguous(&uniform.axes()[1], &[65, 91, 127]);
        assert_contiguous(&uniform.axes()[1], &[2047, 2048, 2049]);

        let causal = DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::CausalTail));
        assert_contiguous(&causal.axes()[0], &[1, 2, 3, 4, 6, 8, 11]);
        assert_contiguous(&causal.axes()[0], &[127, 128, 129, 130, 131, 132, 133]);
        assert_contiguous(&causal.axes()[0], &[254, 255, 256, 257, 258, 263, 264, 265]);
        assert_contiguous(&causal.axes()[1], &[127, 128, 129, 130, 182, 200]);
        assert_contiguous(&causal.axes()[1], &[255, 256, 260, 511, 512, 513, 724]);
        assert_contiguous(&causal.axes()[1], &[2047, 2048, 2049]);

        let speculative =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::SpeculativeGroups {
                group_size: 2,
            }));
        assert!(speculative.axes()[0]
            .iter()
            .all(|query| *query as u32 % 2 == 0));
        assert_contiguous(&speculative.axes()[0], &[128, 132, 134]);
        assert_contiguous(&speculative.axes()[0], &[256, 264, 266]);
        assert_contiguous(&speculative.axes()[1], &[16, 23, 32, 45, 63]);
        assert_contiguous(&speculative.axes()[1], &[129, 182, 256]);
        assert_contiguous(&speculative.axes()[1], &[2047, 2048, 2049]);
    }

    #[test]
    fn masks_match_each_frozen_pattern_domain() {
        let uniform_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::UniformFull));
        let uniform = mask_for(ValidCountsPattern::UniformFull, &uniform_grid);
        assert_mask_split(&uniform, 810, 0);

        let causal_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::CausalTail));
        let causal = mask_for(ValidCountsPattern::CausalTail, &causal_grid);
        assert_mask_split(&causal, 858, 659);
        assert!(!masked(&causal, &causal_grid, 3, 3));
        assert!(!masked(&causal, &causal_grid, 3, 4));
        assert!(masked(&causal, &causal_grid, 3, 2));
        assert!(!masked(&causal, &causal_grid, 132, 182));
        assert!(masked(&causal, &causal_grid, 200, 182));
        assert!(!masked(&causal, &causal_grid, 255, 255));
        assert!(masked(&causal, &causal_grid, 256, 255));
        assert!(!masked(&causal, &causal_grid, 4096, 131072));
        assert!(!masked(&causal, &causal_grid, 65536, 1_048_576));
        assert!(masked(&causal, &causal_grid, 65536, 32768));

        let speculative_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::SpeculativeGroups {
                group_size: 2,
            }));
        let speculative = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &speculative_grid,
        );
        assert_mask_split(&speculative, 620, 0);
        assert!(!masked(&speculative, &speculative_grid, 2, 1));
        assert!(!masked(&speculative, &speculative_grid, 32, 2048));
        assert!(!masked(&speculative, &speculative_grid, 134, 65536));
        assert!(!masked(&speculative, &speculative_grid, 266, 65536));
        assert!(!masked(&speculative, &speculative_grid, 4096, 131072));
        assert!(!masked(&speculative, &speculative_grid, 65536, 1_048_576));
    }

    #[test]
    fn trtllm_fp8_masks_only_the_rejected_65536_query_row() {
        let mut cfg = config(ValidCountsPattern::CausalTail);
        cfg.backends = vec!["flashinfer_trtllm_fp8"];
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);

        assert!(!masked(&mask, &grid, 32768, 1_048_576));
        assert!(masked(&mask, &grid, 65536, 1_048_576));
    }

    const TRTLLM_BACKEND: &str = "flashinfer_trtllm_fp8";

    /// GLM-5.3-Flash TP4 decode on B200, the exact values of b2-dsa 5.1.
    fn pooled_config() -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![TRTLLM_BACKEND],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: Dim::param("num_attention_heads", 16),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2176,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 0),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            valid_counts_pattern: ValidCountsPattern::PooledUniformFull {
                index_topk: 2048,
                index_kpool: 4,
            },
            index_distribution: "unique_scattered_pages".to_string(),
            cache_layout: "hnd_paged_mqa_fp8_latent".to_string(),
        }
    }

    /// Catches a pooled decode clipping at the 2176-wide page table (up to 125
    /// phantom active slots per row) or any drift from the rope-free runner args.
    #[test]
    fn pooled_payloads_cap_at_topk_plus_kpool_minus_one() {
        let cfg = pooled_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, TRTLLM_BACKEND);

        // Past index_topk the rows spread over the four tail phases (job 1141).
        for (q, s, expected) in [
            (32, 1024, "u:1024x32"),
            (32, 2048, "u:2048x32"),
            (32, 2051, "g:(2048,2049,2050,2051)x8"),
            (32, 4096, "g:(2048,2049,2050,2051)x8"),
            (4, 4096, "r:2048..2051"),
            (2, 4096, "r:2048..2049"),
        ] {
            let fields = payload_for(&payloads, &grid, q, s).fields();
            assert_eq!(fields.get("valid_counts"), Some(&Value::from(expected)));
        }
        let odd = payload_for(&payloads, &grid, 127, 4096).fields()["valid_counts"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(odd.starts_with("g:(2048,2049,2050,2051,2048,") && odd.ends_with(",2050)x1"));
        let fields = payload_for(&payloads, &grid, 1, 1_048_576).fields();
        assert_eq!(fields.get("valid_counts"), Some(&Value::from("u:2048x1")));
        assert_eq!(
            serde_json::to_value(fields).unwrap(),
            serde_json::json!({
                "backend": TRTLLM_BACKEND,
                "num_queries": 1,
                "num_cache_tokens": 1_048_576,
                "num_heads": 16,
                "num_kv_heads": 1,
                "selected_k": 2176,
                "latent_dim": 512,
                "rope_dim": 0,
                "value_dim": 512,
                "softmax_scale": 0.0625,
                "q_dtype": "fp8_e4m3",
                "cache_dtype": "fp8_e4m3",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts": "u:2048x1",
                "index_distribution": "unique_scattered_pages",
                "cache_layout": "hnd_paged_mqa_fp8_latent",
            })
        );
        assert_eq!(cfg.compute_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }

    /// Catches the pooled grid losing the saturation bracket or growing past
    /// the 500-feasible-coordinate ceiling.
    #[test]
    fn pooled_grid_brackets_saturation_within_the_coordinate_ceiling() {
        let cfg = pooled_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        assert_contiguous(&grid.axes()[1], &[1024, 1536, 2048, 2051, 4096]);
        assert_eq!(grid.axes()[0].last(), Some(&32768.0));
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        assert_mask_split(&mask, 21 * 23, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds selected_k")]
    fn pooled_cap_must_fit_the_page_table() {
        let mut cfg = pooled_config();
        cfg.selected_k = 2048;
        DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
    }

    /// Catches the pooled pattern escaping the TRTLLM >32,768-query guard.
    #[test]
    fn pooled_pattern_keeps_the_trtllm_query_guard() {
        let pooled = pooled_config();
        let synthetic = crate::timing::sweep::SweepGrid::new(vec![
            crate::timing::sweep::Axis::values([32768, 65536]),
            crate::timing::sweep::Axis::values([2051]),
        ]);
        assert_eq!(
            DsaSparseMlaAttentionSpec::infeasible_mask(&pooled, &synthetic),
            vec![false, true]
        );
    }

    /// GLM-5.3-Flash TP4 on H200 under vLLM v0.31 (`FLASH_ATTN_MLA_SPARSE`).
    fn fa3_config() -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![FA3_SPARSE_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            q_dtype: DType::Bf16,
            cache_dtype: DType::Bf16,
            cache_layout: "token_major_mqa_bf16_latent".to_string(),
            ..pooled_config()
        }
    }

    /// Catches the FA3 grid losing the measured Combine breakpoint (a cache
    /// point on each side of Q 992 | 993, so no integer Q interpolates across
    /// it) or growing past the 500-coordinate ceiling.
    #[test]
    fn fa3_grid_pins_the_combine_breakpoint_within_the_ceiling() {
        let cfg = fa3_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        assert_eq!(FA3_COMBINE_MAX_QUERIES, 992);
        assert_contiguous(&grid.axes()[0], &[768, 992, 993, 1024]);
        assert_contiguous(&grid.axes()[1], &[1536, 2048, 2051, 4096]);
        assert_eq!(grid.axes()[0].last(), Some(&32768.0));
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        assert_mask_split(&mask, 23 * 20, 0);
    }

    /// Catches an FA3 payload that drifts from the Python runner's accepted
    /// rope-free BF16 latent layout.
    #[test]
    fn fa3_payloads_are_rope_free_bf16_latent_rows() {
        let cfg = fa3_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FA3_SPARSE_BACKEND);
        assert_eq!(payloads.len(), 460);
        let mut fields =
            serde_json::to_value(payload_for(&payloads, &grid, 992, 65536).fields()).unwrap();
        // 992 rows spread evenly over the four kpool tail phases.
        assert_eq!(fields["valid_counts"], "g:(2048,2049,2050,2051)x248");
        fields["valid_counts"] = Value::from("checked above");
        assert_eq!(
            fields,
            serde_json::json!({
                "backend": FA3_SPARSE_BACKEND,
                "num_queries": 992,
                "num_cache_tokens": 65536,
                "num_heads": 16,
                "num_kv_heads": 1,
                "selected_k": 2176,
                "latent_dim": 512,
                "rope_dim": 0,
                "value_dim": 512,
                "softmax_scale": 0.0625,
                "q_dtype": "bf16",
                "cache_dtype": "bf16",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts": "checked above",
                "index_distribution": "unique_scattered_pages",
                "cache_layout": "token_major_mqa_bf16_latent",
            })
        );
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Bf16));
    }

    #[test]
    #[should_panic(expected = "cannot share a config")]
    fn fa3_does_not_share_a_grid_with_another_backend() {
        let mut cfg = fa3_config();
        cfg.backends = vec![FA3_SPARSE_BACKEND, TRTLLM_BACKEND];
        DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
    }

    /// GLM-5.3-Flash TP4 on H200 under `FLASHINFER_MLA_SPARSE_SM90`.
    fn flashinfer_sm90_config() -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![FLASHINFER_SM90_SPARSE_BACKEND],
            ..fa3_config()
        }
    }

    /// Catches a FlashInfer grid that would let the linear cache interpolate
    /// across a planner wave step (a query point off a multiple of 132 above
    /// one wave), sample contexts on the fast full-tile teeth, drop the dense
    /// sub-wave points, or pass 500 coordinates.
    #[test]
    fn flashinfer_sm90_grid_samples_whole_waves_within_the_ceiling() {
        let cfg = flashinfer_sm90_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let queries = &grid.axes()[0];
        assert!(queries
            .iter()
            .filter(|&&q| q > 132.0)
            .all(|&q| q as u32 % 132 == 0));
        assert_contiguous(queries, &[112, 128, 132, 264, 396]);
        // Partial-tile contexts (64k + 40) below the cap, then the cap itself.
        assert_contiguous(&grid.axes()[1], &[40, 104, 168, 296]);
        assert_contiguous(&grid.axes()[1], &[1576, 2024, 2051, 4096]);
        // The planner's 16384-work capacity bounds the grid.
        assert_eq!(queries.last(), Some(&16368.0));
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        assert_mask_split(&mask, 25 * 19, 0);
    }

    /// Catches a projection that would price 133 rows like 132 (a whole wave
    /// short), round inside the first wave, or round the context axis.
    #[test]
    fn flashinfer_sm90_cache_coords_round_queries_up_to_the_wave() {
        let cfg = flashinfer_sm90_config();
        let coords = |num_queries, num_cache_tokens| {
            let input = DsaSparseMlaAttentionKernelInput {
                num_queries,
                num_cache_tokens,
            };
            DsaSparseMlaAttentionSpec::cache_coords(&cfg, &input).to_vec()
        };
        assert_eq!(coords(100, 3000), vec![100.0, 3000.0]);
        assert_eq!(coords(132, 3000), vec![132.0, 3000.0]);
        assert_eq!(coords(133, 3000), vec![264.0, 3000.0]);
        assert_eq!(coords(384, 777), vec![396.0, 777.0]);
        assert_eq!(coords(7892, 1037), vec![7920.0, 1037.0]);
        // FA3 keeps its physical axes.
        let input = DsaSparseMlaAttentionKernelInput {
            num_queries: 133,
            num_cache_tokens: 3000,
        };
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_coords(&fa3_config(), &input).to_vec(),
            vec![133.0, 3000.0]
        );
    }

    /// Catches a FlashInfer payload that drifts from FA3's (the two backends
    /// share the kind's args: only `backend` differs) or a GPU whose planner
    /// waves are unknown being accepted.
    #[test]
    fn flashinfer_sm90_payloads_match_fa3_and_need_a_known_sm_count() {
        let cfg = flashinfer_sm90_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads =
            DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHINFER_SM90_SPARSE_BACKEND);
        assert_eq!(payloads.len(), 25 * 19);
        let mut fields =
            serde_json::to_value(payload_for(&payloads, &grid, 264, 65536).fields()).unwrap();
        assert_eq!(fields["valid_counts"], "g:(2048,2049,2050,2051)x66");
        assert_eq!(fields["backend"], FLASHINFER_SM90_SPARSE_BACKEND);
        fields["backend"] = Value::from(FA3_SPARSE_BACKEND);
        let fa3 = fa3_config();
        let fa3_grid = DsaSparseMlaAttentionSpec::sweep_grid(&fa3);
        let fa3_payloads =
            DsaSparseMlaAttentionSpec::enumerate(&fa3, &fa3_grid, FA3_SPARSE_BACKEND);
        let mut expected =
            serde_json::to_value(payload_for(&fa3_payloads, &fa3_grid, 256, 65536).fields())
                .unwrap();
        expected["num_queries"] = Value::from(264);
        expected["valid_counts"] = Value::from("g:(2048,2049,2050,2051)x66");
        assert_eq!(fields, expected);
        assert!(DsaSparseMlaAttentionSpec::validate_config(&cfg).is_ok());
        let mut h100 = cfg.clone();
        h100.gpu_name = "NVIDIA H100".into();
        assert!(DsaSparseMlaAttentionSpec::validate_config(&h100).is_err());
        assert!(DsaSparseMlaAttentionSpec::validate_config(&fa3).is_ok());
    }

    /// Catches the FP8-cache rows drifting from the BF16 FlashInfer rows in
    /// anything but the cache identity (the 475 profiled coordinates must be
    /// the same planner-wave grid, and the payload must say e4m3 latent rows),
    /// or the two backends being mixed in one config.
    #[test]
    fn flashinfer_sm90_fp8_rows_share_the_grid_and_change_only_the_cache() {
        let bf16 = flashinfer_sm90_config();
        let fp8 = DsaSparseMlaAttentionKernelConfig {
            backends: vec![FLASHINFER_SM90_SPARSE_FP8_BACKEND],
            cache_dtype: DType::Fp8E4m3,
            cache_layout: "token_major_mqa_fp8_latent".into(),
            ..bf16.clone()
        };
        assert_eq!(fp8.kv_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(fp8.compute_dtype(), Some(DType::Bf16));
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&fp8);
        assert_eq!(grid, DsaSparseMlaAttentionSpec::sweep_grid(&bf16));
        let input = DsaSparseMlaAttentionKernelInput {
            num_queries: 133,
            num_cache_tokens: 3000,
        };
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_coords(&fp8, &input).to_vec(),
            vec![264.0, 3000.0]
        );
        let payloads =
            DsaSparseMlaAttentionSpec::enumerate(&fp8, &grid, FLASHINFER_SM90_SPARSE_FP8_BACKEND);
        assert_eq!(payloads.len(), 475);
        let bf16_payloads =
            DsaSparseMlaAttentionSpec::enumerate(&bf16, &grid, FLASHINFER_SM90_SPARSE_BACKEND);
        let mut fields =
            serde_json::to_value(payload_for(&payloads, &grid, 264, 2051).fields()).unwrap();
        assert_eq!(fields["cache_dtype"], "fp8_e4m3");
        assert_eq!(fields["cache_layout"], "token_major_mqa_fp8_latent");
        fields["backend"] = Value::from(FLASHINFER_SM90_SPARSE_BACKEND);
        fields["cache_dtype"] = Value::from("bf16");
        fields["cache_layout"] = Value::from("token_major_mqa_bf16_latent");
        assert_eq!(
            fields,
            serde_json::to_value(payload_for(&bf16_payloads, &grid, 264, 2051).fields()).unwrap()
        );
        let mut h100 = fp8.clone();
        h100.gpu_name = "NVIDIA H100".into();
        assert!(DsaSparseMlaAttentionSpec::validate_config(&h100).is_err());
        let mixed = DsaSparseMlaAttentionKernelConfig {
            backends: vec![
                FLASHINFER_SM90_SPARSE_BACKEND,
                FLASHINFER_SM90_SPARSE_FP8_BACKEND,
            ],
            ..fp8
        };
        assert!(
            std::panic::catch_unwind(|| DsaSparseMlaAttentionSpec::sweep_grid(&mixed)).is_err()
        );
    }

    #[test]
    fn speculative_odd_query_defense_remains_for_synthetic_grids() {
        let grid = crate::timing::sweep::SweepGrid::new(vec![
            crate::timing::sweep::Axis::values([2, 3, 4]),
            crate::timing::sweep::Axis::values([1, 2]),
        ]);
        let mask = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &grid,
        );

        assert_mask_split(&mask, 4, 2);
        assert!(masked(&mask, &grid, 3, 1));
        assert!(masked(&mask, &grid, 3, 2));
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                3,
                2,
                2048
            ),
            "u:0x3"
        );
    }

    #[test]
    fn canonical_valid_counts_obeys_python_precedence_for_every_pattern() {
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::UniformFull, 1, 1, 2048),
            "u:1x1"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::UniformFull, 256, 131072, 2048),
            "u:2048x256"
        );

        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 1, 1, 2048),
            "u:1x1"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 128, 2048),
            "r:1..128"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 2048, 2048),
            "r:1921..2048"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 2049, 2048),
            "c:1922..2049@2048"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 4096, 2048),
            "u:2048x128"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 4096, 1, 2048),
            "u:0x4096"
        );

        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                2,
                1,
                2048
            ),
            "r:0..1"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                32,
                2048,
                2048
            ),
            "g:(2047,2048)x16"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                2,
                2049,
                2048
            ),
            "u:2048x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                127,
                2048,
                2048
            ),
            "u:0x127"
        );

        // Group six, at coordinates its own grid actually visits. `r:`/`c:` need
        // a single repeat, so they only appear at the smallest query anchor.
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                5,
                2048
            ),
            "r:0..5"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                2043,
                2048
            ),
            "r:2038..2043"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                2049,
                2048
            ),
            "c:2044..2049@2048"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                3,
                2048
            ),
            "g:(0,0,0,1,2,3)x1"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                2048,
                2048
            ),
            "g:(2043,2044,2045,2046,2047,2048)x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                2049,
                2048
            ),
            "g:(2044,2045,2046,2047,2048,2048)x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                4096,
                2048
            ),
            "u:2048x12"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                7,
                4096,
                2048
            ),
            "u:0x7"
        );
    }

    /// Group two is a measured grid, not just an encoding. Generalizing the
    /// derivation must leave both its axes and its RLE bytes exactly where the
    /// R.4 gate put them.
    #[test]
    fn group_two_keeps_its_frozen_axes_and_bytes() {
        assert_eq!(speculative_query_axis(2), SPECULATIVE_QUERY_AXIS);
        assert_eq!(speculative_cache_axis(2, 2048), SPECULATIVE_CACHE_AXIS);

        // The generalized encoder must reproduce the frozen group-two bytes on
        // every coordinate of the frozen grid, not just the sampled ones.
        for &num_queries in SPECULATIVE_QUERY_AXIS.iter() {
            for &num_cache_tokens in SPECULATIVE_CACHE_AXIS.iter() {
                let first = num_cache_tokens.saturating_sub(1).min(2048);
                let second = num_cache_tokens.min(2048);
                let expected = if first == second {
                    format!("u:{first}x{num_queries}")
                } else if num_queries == 2 {
                    format!("r:{first}..{second}")
                } else {
                    format!("g:({first},{second})x{}", num_queries / 2)
                };
                assert_eq!(
                    canonical_valid_counts(
                        ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                        num_queries,
                        num_cache_tokens,
                        2048
                    ),
                    expected,
                    "group-two identity drifted at ({num_queries}, {num_cache_tokens})"
                );
            }
        }
    }

    /// A wider verify group changes RLE shape where the group first fits the
    /// context and where it straddles the `selected_k` clip. Anchoring the query
    /// axis on request count keeps every cell feasible; dropping either cache
    /// boundary leaves the cache interpolating across a discontinuity.
    #[test]
    fn group_six_grid_anchors_requests_and_pins_its_transitions() {
        let cfg = config(ValidCountsPattern::SpeculativeGroups { group_size: 6 });
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);

        assert_eq!(grid.axes()[0].len(), SPECULATIVE_QUERY_AXIS.len());
        assert_eq!(grid.axes()[0][0], 6.0);
        assert_eq!(grid.axes()[0].last(), Some(&196_608.0));
        assert!(grid.axes()[0].iter().all(|q| *q as u32 % 6 == 0));
        for (index, &num_queries) in SPECULATIVE_QUERY_AXIS.iter().enumerate() {
            assert_eq!(grid.axes()[0][index], f64::from(num_queries / 2 * 6));
        }

        // Retain existing points and add both sides of the all-rows-saturated boundary.
        assert_eq!(grid.axes()[1].len(), SPECULATIVE_CACHE_AXIS.len() + 7);
        assert_contiguous(&grid.axes()[1], &[4, 5, 6, 7, 8]);
        assert_contiguous(&grid.axes()[1], &[1024, 2043, 2047, 2048, 2049]);
        assert_contiguous(&grid.axes()[1], &[2049, 2052, 2053, 2054, 4096]);
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));

        let mask = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 6 },
            &grid,
        );
        assert_mask_split(&mask, 760, 0);

        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);
        assert_eq!(payloads.len(), 760);
        assert_payload(
            payload_for(&payloads, &grid, 12, 2052),
            12,
            2052,
            "g:(2047,2048,2048,2048,2048,2048)x2".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 12, 2053),
            12,
            2053,
            "u:2048x12".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 6, 2049),
            6,
            2049,
            "c:2044..2049@2048".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 12, 2043),
            12,
            2043,
            "g:(2038,2039,2040,2041,2042,2043)x2".to_string(),
        );
    }

    #[test]
    fn both_backends_use_the_existing_physical_bilinear_cache() {
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(FLASHMLA_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn enumeration_emits_backend_plus_the_exact_python_schema() {
        for (pattern, expected_payloads) in [
            (ValidCountsPattern::UniformFull, 810),
            (ValidCountsPattern::CausalTail, 1_517),
            (ValidCountsPattern::SpeculativeGroups { group_size: 2 }, 620),
        ] {
            let cfg = config(pattern);
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
            let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

            assert_eq!(payloads.len(), expected_payloads);
            let expected_names = [
                "backend",
                "cache_dtype",
                "cache_layout",
                "index_distribution",
                "index_dtype",
                "latent_dim",
                "num_cache_tokens",
                "num_heads",
                "num_kv_heads",
                "num_queries",
                "output_dtype",
                "q_dtype",
                "rope_dim",
                "selected_k",
                "softmax_scale",
                "valid_counts",
                "value_dim",
            ];
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(names, expected_names);
                assert_eq!(payload.fields().len(), 17);
            }

            let first_q = grid.axes()[0][0] as u32;
            let first_s = grid.axes()[1][0] as u32;
            assert_payload(
                &payloads[0],
                first_q,
                first_s,
                canonical_valid_counts(pattern, first_q, first_s, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 32, 2048),
                32,
                2048,
                canonical_valid_counts(pattern, 32, 2048, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 128, 2049),
                128,
                2049,
                canonical_valid_counts(pattern, 128, 2049, 2048),
            );
            let last_q = *grid.axes()[0].last().unwrap() as u32;
            let last_s = *grid.axes()[1].last().unwrap() as u32;
            assert_payload(
                payloads.last().unwrap(),
                last_q,
                last_s,
                canonical_valid_counts(pattern, last_q, last_s, 2048),
            );
        }

        assert_critical_payloads(
            ValidCountsPattern::UniformFull,
            &[(132, 6), (132, 11), (132, 91), (131, 8192), (133, 8192)],
        );
        assert_critical_payloads(
            ValidCountsPattern::CausalTail,
            &[
                (3, 3),
                (132, 182),
                (32, 511),
                (32, 513),
                (255, 724),
                (130, 130),
                (254, 255),
                (258, 260),
                (4096, 1),
            ],
        );
        assert_critical_payloads(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &[
                (32, 3),
                (132, 23),
                (132, 45),
                (132, 182),
                (134, 65536),
                (266, 65536),
                (4096, 131072),
            ],
        );
    }

    #[test]
    fn remediated_pattern_union_matches_the_frozen_next_phase_inventory() {
        let distributions = [
            "recent_contiguous",
            "unique_scattered_pages",
            "clustered_pages",
            "uniform_stride",
        ];
        let mut all_distributions = BTreeSet::new();
        let mut memberships = 0;

        for distribution in distributions {
            let uniform = feasible_payload_keys(ValidCountsPattern::UniformFull, distribution);
            let causal = feasible_payload_keys(ValidCountsPattern::CausalTail, distribution);
            let speculative = feasible_payload_keys(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                distribution,
            );

            assert_eq!(uniform.len(), 810);
            assert_eq!(causal.len(), 858);
            assert_eq!(speculative.len(), 620);
            assert_eq!(uniform.intersection(&causal).count(), 249);
            assert_eq!(uniform.intersection(&speculative).count(), 180);
            assert_eq!(causal.intersection(&speculative).count(), 168);
            assert_eq!(
                uniform
                    .intersection(&causal)
                    .filter(|payload| speculative.contains(*payload))
                    .count(),
                148
            );

            memberships += uniform.len() + causal.len() + speculative.len();
            let per_distribution: BTreeSet<_> = uniform
                .into_iter()
                .chain(causal)
                .chain(speculative)
                .collect();
            assert_eq!(per_distribution.len(), 1_839);
            all_distributions.extend(per_distribution);
        }

        assert_eq!(memberships, 9_152);
        assert_eq!(all_distributions.len(), 7_356);
    }

    fn rich_dim(name: &str, value: u32) -> Value {
        serde_json::json!({
            "value": value,
            "expression": name,
            "bindings": {name: value},
        })
    }

    fn assert_grid(
        pattern: ValidCountsPattern,
        query_axis: &[u32],
        cache_axis: &[u32],
        cells: usize,
    ) {
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], axis_values(query_axis));
        assert_eq!(axes[1], axis_values(cache_axis));
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(axes[0].first().copied(), Some(f64::from(query_axis[0])));
        assert_eq!(
            axes[0].last().copied(),
            Some(f64::from(*query_axis.last().unwrap()))
        );
        assert_eq!(axes[1].first().copied(), Some(f64::from(cache_axis[0])));
        assert_eq!(
            axes[1].last().copied(),
            Some(f64::from(*cache_axis.last().unwrap()))
        );
        assert_eq!(axes[0].len() * axes[1].len(), cells);
    }

    fn axis_values(axis: &[u32]) -> Vec<f64> {
        axis.iter().map(|&value| f64::from(value)).collect()
    }

    fn assert_contiguous(axis: &[f64], expected: &[u32]) {
        let expected = axis_values(expected);
        assert!(
            axis.windows(expected.len())
                .any(|window| window == expected.as_slice()),
            "axis {axis:?} does not contain contiguous sequence {expected:?}"
        );
    }

    fn assert_critical_payloads(pattern: ValidCountsPattern, coordinates: &[(u32, u32)]) {
        let cfg = config(pattern);
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

        for &(q, s) in coordinates {
            assert_payload(
                payload_for(&payloads, &grid, q, s),
                q,
                s,
                canonical_valid_counts(pattern, q, s, 2048),
            );
        }
    }

    fn feasible_payload_keys(pattern: ValidCountsPattern, distribution: &str) -> BTreeSet<String> {
        let mut cfg = config(pattern);
        cfg.index_distribution = distribution.to_string();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND)
            .into_iter()
            .zip(mask)
            .filter_map(|(payload, masked)| {
                (!masked).then(|| serde_json::to_string(payload.fields()).unwrap())
            })
            .collect()
    }

    fn mask_for(pattern: ValidCountsPattern, grid: &crate::timing::sweep::SweepGrid) -> Vec<bool> {
        DsaSparseMlaAttentionSpec::infeasible_mask(&config(pattern), grid)
    }

    fn assert_mask_split(mask: &[bool], feasible: usize, masked: usize) {
        assert_eq!(mask.len(), feasible + masked);
        assert_eq!(mask.iter().filter(|&&value| !value).count(), feasible);
        assert_eq!(mask.iter().filter(|&&value| value).count(), masked);
    }

    fn masked(mask: &[bool], grid: &crate::timing::sweep::SweepGrid, q: u32, s: u32) -> bool {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        mask[q_index * grid.axes()[1].len() + s_index]
    }

    fn payload_for<'a>(
        payloads: &'a [ArgsPayload],
        grid: &crate::timing::sweep::SweepGrid,
        q: u32,
        s: u32,
    ) -> &'a ArgsPayload {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        &payloads[q_index * grid.axes()[1].len() + s_index]
    }

    fn assert_payload(payload: &ArgsPayload, q: u32, s: u32, valid_counts: String) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(FLASHMLA_BACKEND)));
        assert_eq!(fields.get("num_queries"), Some(&Value::from(q)));
        assert_eq!(fields.get("num_cache_tokens"), Some(&Value::from(s)));
        assert_eq!(fields.get("num_heads"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("selected_k"), Some(&Value::from(2048_u32)));
        assert_eq!(fields.get("latent_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("rope_dim"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("value_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("softmax_scale"), Some(&Value::from(0.0625)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("cache_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("index_dtype"), Some(&Value::from("int32")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("valid_counts"), Some(&Value::from(valid_counts)));
        assert_eq!(
            fields.get("index_distribution"),
            Some(&Value::from("recent_contiguous"))
        );
        assert_eq!(
            fields.get("cache_layout"),
            Some(&Value::from("token_major_mqa_bf16_latent_rope"))
        );
        assert_eq!(payload.backend(), Some(FLASHMLA_BACKEND));
    }
}
