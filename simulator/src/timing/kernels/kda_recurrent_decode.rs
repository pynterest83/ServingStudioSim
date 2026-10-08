//! GLM-5.3-Flash KDA `fused_recurrent_kda` call: plain decode or MTP verify.
//!
//! One semantic call. On the vLLM v0.31 fork it is a single
//! `fused_recurrent_gated_delta_rule_fwd_kernel` launch at every batch: the
//! kernel reads the row-strided q/k/v/beta views of the merged input
//! projection in place. (The 3f667d7 image's older call made four
//! `.contiguous()` copies first; rows it measured carry that image's
//! `backend_version`.)
//!
//! `query_len` is the tokens per request: 1 for a plain decode step, K + 1 for
//! the draft-verify step of K-draft speculative decoding, where each request
//! runs its tokens serially and stores a state snapshot per token. It is
//! static identity like the head geometry and dtype; decode batch size
//! (requests) is the only runtime interpolation axis. The fp32 recurrent state
//! is fixed by the model, so it is not a key.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecurrentDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Tokens per request: 1 (plain decode) or K + 1 (verify K drafts).
    pub query_len: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KdaRecurrentDecodeKernelInput {
    pub batch_size: u32,
}

pub struct KdaRecurrentDecodeSpec;

impl KernelSpec for KdaRecurrentDecodeSpec {
    type Config = KdaRecurrentDecodeKernelConfig;
    type Input = KdaRecurrentDecodeKernelInput;

    const KIND: KernelKind = "kda_recurrent_decode";

    /// Batch 1..512 in powers of two plus 12 and 24, for every `query_len`.
    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Powers of two plus 12 and 24: on the 3f667d7 image the batch-8..32
        // curve was concave (B200 job 1138), so pow2 alone missed batch 12/24
        // by 11-16%. The grid is kept so existing rows stay on it.
        let mut batches = Axis::pow2(0, 9);
        batches.extend([12.0, 24.0]);
        batches.sort_by(f64::total_cmp);
        SweepGrid::new(vec![batches])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|batch_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("query_len", config.query_len)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(KdaRecurrentDecodeKernel, KdaRecurrentDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput, KdaRecurrentDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> KdaRecurrentDecodeKernelConfig {
        KdaRecurrentDecodeKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            query_len: 1,
            num_heads: 16.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a dtype tag drift that would hide the kind from launcher
    /// backend selection (Python BackendSupport is compute bf16, no KV axis).
    #[test]
    fn dtype_tag_is_compute_only() {
        let cfg = config();
        assert_eq!(KdaRecurrentDecodeSpec::KIND, "kda_recurrent_decode");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    /// Catches a grid that drops the batch-1 (no-copy) point or the measured
    /// 12/24 curvature points, or grows past its 12-point feasible budget.
    #[test]
    fn grid_is_pow2_1_to_512_plus_12_and_24_with_twelve_feasible_points() {
        let cfg = config();
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        );
        assert!(KdaRecurrentDecodeSpec::infeasible_mask(&cfg, &grid).is_empty());
        assert_eq!(
            KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton").len(),
            12
        );
        assert_eq!(
            KdaRecurrentDecodeSpec::cache_kind("vllm_triton"),
            CacheKind::Cache1DLinear
        );
    }

    /// Catches payload keys that would miss the Python KernelArgs DB key.
    #[test]
    fn enumerate_emits_exact_python_fields() {
        let cfg = config();
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        let payloads = KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "batch_size",
                    "dtype",
                    "head_dim",
                    "num_heads",
                    "query_len"
                ]
            );
            assert_eq!(payload.backend(), Some("vllm_triton"));
        }
        let first = payloads[0].fields();
        assert_eq!(first["batch_size"], Value::from(1_u32));
        assert_eq!(first["num_heads"], Value::from(16_u32));
        assert_eq!(first["head_dim"], Value::from(128_u32));
        assert_eq!(first["dtype"], Value::from("bf16"));
        assert_eq!(first["query_len"], Value::from(1_u32));
        assert_eq!(payloads[11].fields()["batch_size"], Value::from(512_u32));
    }

    /// Catches a verify width that changes the sweep instead of riding along as
    /// static identity, or never reaches the Python `query_len` key (a verify
    /// leaf would then read plain-decode rows).
    #[test]
    fn query_len_is_static_identity_forwarded_to_every_payload() {
        let cfg = KdaRecurrentDecodeKernelConfig {
            query_len: 6,
            ..config()
        };
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes(),
            KdaRecurrentDecodeSpec::sweep_grid(&config()).axes()
        );
        let payloads = KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 12);
        for payload in &payloads {
            assert_eq!(payload.fields()["query_len"], Value::from(6_u32));
        }
        assert_ne!(cfg, config());
    }

    /// Catches an Input that does not project batch_size onto the cache axis
    /// or does not reach the cost-log slot payload.
    #[test]
    fn input_projects_batch_size_and_logs_it() {
        let input = KdaRecurrentDecodeKernelInput { batch_size: 24 };
        assert_eq!(&*input.coords(), &[24.0]);
        assert_eq!(
            KdaRecurrentDecodeKernelInput::coord_field_names(),
            &["batch_size"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 24})
        );
    }
}
