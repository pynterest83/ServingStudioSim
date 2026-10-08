//! FlashInfer prefill (causal) attention kernel: one cached perf model per
//! attention-dims config.
//!
//! Everything generic (build / eval / the `Probe` impl / for-backend loops)
//! lives in `engine::Kernel<S>`. This file declares the prefill-specific Config /
//! Input, the `KIND` wire string, and the `enumerate` body that lifts (config,
//! sweep coord, backend) to the on-wire `ArgsPayload`. The Python
//! `FlashinferAttnPrefillArgs` dataclass owns the matching schema.
//!
//! **Cache dims are not the kernel params, and not the query interface.** A
//! caller queries the kernel with the physical history-append shape
//! `(prefix_len = k, append_len = q)` (`prefix_len == 0` is fresh prefill); the
//! Python runner derives `q_len = append_len`, `kv_len = prefix_len + append_len`.
//! But the *cache* is built over the re-axis `A = k + q/2`, `B = q`. Causal
//! per-iteration work is `q·(k + q/2) = A·B`, which is *exactly* bilinear in
//! `(A, B)` — so `Cache2DLinear` interp AND extrapolation carry no residual
//! `q²/2` curvature. The Input's `SweepCoords` does the `(k,q) → (A,B)`
//! projection, so the re-axis is invisible to callers.
//! (See `[[attn-quadratic-interp-unsuitable]]` and
//! `agent-trace/flashinfer_attn_prefill_reaxis.md`.)
//!
//! `enumerate` maps each `(A,B)` grid cell back to the physical shape to profile
//! (`append = B`, `prefix = A − B/2`). The `A < B/2` corner of the rectangular
//! grid is physically unreachable (`A ≥ B/2` always); `infeasible_mask` strips
//! those cells so the cache drops + renormalizes rather than fabricating data.
//!
//! `rect` is the non-causal sibling (`flashinfer_attn_rect.rs`); it has no `q²/2`
//! triangle, so it keeps physical axes and does NOT use this re-axis.
//!
//! 2D note: two monotonic axes → `Cache2DLinear` + `grid.expand_2d`. See
//! `timing/sweep.rs`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

/// External query schema is the physical history-append shape; a manual
/// `SweepCoords` projects it to the `(A, B)` axes the cache is built on.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnPrefillKernelInput {
    pub prefix_len: u32,
    pub append_len: u32,
}

impl SweepCoords for FlashinferAttnPrefillKernelInput {
    fn coords(&self) -> Coords {
        // (k, q) → (k + q/2, q). `A` carries the causal triangle so `A·B` is the
        // full per-iteration work with no quadratic residual for bilinear to miss.
        let a = self.prefix_len as f64 + self.append_len as f64 / 2.0;
        Coords::new([a, self.append_len as f64])
    }
    fn coord_field_names() -> &'static [&'static str] {
        &["effective_kv_tokens", "append_len"]
    }
}

/// vLLM v0.31's own FA3 varlen call for MLA dense-MHA prefill new tokens. It
/// profiles `prefix_len == 0` only, so it cannot share the (A, B) grid with the
/// FlashInfer backends and must not share a config with them.
pub const VLLM_FA3_BACKEND: &str = "vllm_fa3";

/// `append_len` points for `vllm_fa3` (with `prefix_len == 0`). On H200 at
/// 16 heads x 256 dims, FA3 splits KV and adds a `FlashAttnFwdCombine` launch
/// exactly for 321..=768 tokens, so 320/321 and 768/769 are adjacent pairs:
/// no integer query falls between them, and the cache never interpolates across
/// either cliff. Production's dense path keeps each request within index_topk
/// (2,048); the equal-work batch mapping can reach 4,096, and 8,192 is headroom.
const VLLM_FA3_APPEND_AXIS: [u32; 29] = [
    1, 16, 32, 64, 128, 192, 256, 320, 321, 384, 448, 512, 576, 640, 704, 768, 769, 832, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 4096, 6144, 8192,
];

/// The `vllm_fa3` grid on the (A, B) cache axes: B = append_len, and A = B/2,
/// the only A a fresh-prefill shape has. Only the diagonal is feasible
/// (`infeasible_mask`); a query between two diagonal points blends them through
/// the cache's renormalized-corner path.
fn vllm_fa3_sweep_grid() -> SweepGrid {
    let appends = Axis::values(VLLM_FA3_APPEND_AXIS);
    let effective = appends.iter().map(|append| append / 2.0).collect();
    SweepGrid::new(vec![effective, appends])
}

pub struct FlashinferAttnPrefillSpec;

impl KernelSpec for FlashinferAttnPrefillSpec {
    type Config = FlashinferAttnPrefillKernelConfig;
    type Input = FlashinferAttnPrefillKernelInput;

    const KIND: KernelKind = "flashinfer_attn_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if config.backends.contains(&VLLM_FA3_BACKEND) {
            return vllm_fa3_sweep_grid();
        }
        // Cache axes are (A, B), NOT physical (prefix, append):
        //   A = prefix_len + append_len/2 — spans the feasible range (min q/2=64
        //     at k=0, up to k_max + q_max/2) with headroom: pow2(6,16) = 64..65536.
        //   B = append_len — a graduated ramp denser than pow2 in the low-q range,
        //     then the log2 tail: 128-step to 1024, 256-step 1024..2048, 512-step
        //     2048..4096, then pow2 to 32768. The dense segments bracket the mid-q
        //     GPU efficiency troughs (fa3 utilization dips ~25% mid-cell, a
        //     wave/CTA-quantization effect) that wide pow2 cells would overshoot.
        SweepGrid::new(vec![
            Axis::pow2(6, 16),
            Axis::chain([
                Axis::arithmetic(128, 1024, 128),
                Axis::arithmetic(1024, 2048, 256),
                Axis::arithmetic(2048, 4096, 512),
                Axis::pow2(7, 15),
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // The (A, B) re-axis above exists precisely so causal work is `A*B`, so
        // the bilinear cross term is this kernel's physics and extends correctly.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        if config.backends.contains(&VLLM_FA3_BACKEND) && config.backends.len() > 1 {
            anyhow::bail!(
                "{VLLM_FA3_BACKEND} profiles prefix_len 0 only and cannot share a config \
                 with {:?}",
                config.backends
            );
        }
        Ok(())
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        if config.backends.contains(&VLLM_FA3_BACKEND) {
            // Fresh prefill only: A == B/2 exactly (prefix_len == 0).
            return grid.expand_2d(|a, b| a != b / 2.0);
        }
        // `A = k + q/2 >= q/2 = B/2` for any real shape, so the `A < B/2` corner
        // of the rectangular (A,B) grid is physically unreachable. The build path
        // strips these cells before profiling and slots a non-finite placeholder
        // back, so the cache drops them and renormalizes over the feasible
        // corners — rather than fabricating a clamped `k=0` time at a low-A node
        // (which would bias fresh-prefill interp high).
        grid.expand_2d(|a, b| a < b / 2.0)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|a, b| {
            // Map the (A,B) grid cell back to the physical shape to profile:
            // append = B, prefix = A - B/2. Feasible cells have A >= B/2 so this
            // is non-negative; the `.max(0.0)` is a guard for the infeasible
            // A < B/2 cells, which `infeasible_mask` strips before profiling.
            let append_len = b.round() as u32;
            let prefix_len = (a - b / 2.0).max(0.0).round() as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("prefix_len", prefix_len)
                .with("append_len", append_len)
        })
    }
}

register_kernel!(FlashinferAttnPrefillKernel, FlashinferAttnPrefillSpec);

#[cfg(test)]
mod tests {
    use super::{
        FlashinferAttnPrefillKernelConfig, FlashinferAttnPrefillKernelInput,
        FlashinferAttnPrefillSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> FlashinferAttnPrefillKernelConfig {
        FlashinferAttnPrefillKernelConfig {
            backends: vec!["fa2", "fa3", "trt", "cudnn"],
            gpu_name: "H100".to_string(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_includes_backends_and_attention_dims() {
        let cfg = config();
        assert_eq!(cfg.backends, vec!["fa2", "fa3", "trt", "cudnn"]);
        assert_eq!(cfg.num_qo_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.q_dtype, DType::Bf16);
        assert_eq!(cfg.kv_dtype, DType::Bf16);
        assert_eq!(cfg.o_dtype, DType::Bf16);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["fa2", "fa3", "trt", "cudnn"], "gpu_name": "H100",
                "num_qo_heads": {"value": 32, "expression": null, "bindings": {}},
                "num_kv_heads": {"value": 8, "expression": null, "bindings": {}},
                "head_dim": {"value": 128, "expression": null, "bindings": {}},
                "q_dtype": "bf16", "kv_dtype": "bf16", "o_dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_coords_project_physical_k_q_to_a_b() {
        // (k=2048, q=512) -> A = k + q/2 = 2304, B = q = 512.
        let input = FlashinferAttnPrefillKernelInput {
            prefix_len: 2048,
            append_len: 512,
        };
        assert_eq!(&*input.coords(), &[2304.0, 512.0]);
    }

    #[test]
    fn coord_field_names_label_the_cache_axes() {
        assert_eq!(
            FlashinferAttnPrefillKernelInput::coord_field_names(),
            ["effective_kv_tokens", "append_len"]
        );
    }

    #[test]
    fn input_deserializes_from_json_for_kernel_query() {
        // `kernel-query` deserializes a query point straight into the kernel's
        // own Input (which `CacheProbe::eval_json` then evals); confirm the field
        // names match what the harness sends, and the (A,B) projection runs.
        let input: FlashinferAttnPrefillKernelInput =
            serde_json::from_str(r#"{"prefix_len":2048,"append_len":512}"#).unwrap();
        assert_eq!(&*input.coords(), &[2304.0, 512.0]);
    }

    #[test]
    fn config_deserializes_from_wire_json_for_kernel_query() {
        // `kernel-query` deserializes the whole config from JSON: backends arrive
        // as owned strings (leaked to `&'static str` via `de_backends`) and dtypes
        // as the snake_case wire literal (`"bf16"`, via DType's hand-written
        // `Deserialize`, NOT the variant `"Bf16"`). Confirm it builds a valid Config.
        let cfg: FlashinferAttnPrefillKernelConfig = serde_json::from_str(
            r#"{"backends":["fa2","fa3"],"gpu_name":"H200","num_qo_heads":32,
                "num_kv_heads":8,"head_dim":128,"q_dtype":"bf16","kv_dtype":"bf16",
                "o_dtype":"bf16"}"#,
        )
        .unwrap();
        assert_eq!(cfg.backends, vec!["fa2", "fa3"]);
        assert_eq!(cfg.gpu_name, "H200");
        assert_eq!(cfg.num_qo_heads, 32);
        assert_eq!(cfg.q_dtype, DType::Bf16);
    }

    #[test]
    fn cache_kind_is_two_dimensional_linear() {
        assert_eq!(
            FlashinferAttnPrefillSpec::cache_kind("fa2"),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }

    #[test]
    fn sweep_grid_is_a_b_axes_with_dense_low_q_segment() {
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 2);
        // A axis = pow2(6,16) = 64..65536 (11 points).
        assert_eq!(grid.axes()[0][0], 64.0);
        assert_eq!(*grid.axes()[0].last().unwrap(), 65536.0);
        assert_eq!(grid.axes()[0].len(), 11);
        // B axis = graduated ramp then log2 tail: 128-step to 1024 (8), 256-step
        // 1024..2048 (+3: 1280,1536,1792,2048), 512-step 2048..4096 (+3:
        // 2560,3072,3584,4096), pow2 tail (+3: 8192,16384,32768) = 19.
        assert_eq!(grid.axes()[1][0], 128.0);
        assert_eq!(*grid.axes()[1].last().unwrap(), 32768.0);
        assert_eq!(grid.axes()[1].len(), 19);
        // The dense points bracketing the mid-q efficiency troughs.
        for q in [384.0, 640.0, 768.0, 896.0, 1280.0, 1792.0, 2560.0, 3584.0] {
            assert!(grid.axes()[1].contains(&q), "B axis must include {q}");
        }
    }

    #[test]
    fn infeasible_mask_drops_a_lt_half_b_cells() {
        let cfg = config();
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&cfg);
        let mask = FlashinferAttnPrefillSpec::infeasible_mask(&cfg, &grid);
        // One bool per (A,B) grid cell, row-major.
        let (a_ax, b_ax) = (&grid.axes()[0], &grid.axes()[1]);
        assert_eq!(mask.len(), a_ax.len() * b_ax.len());
        // A cell is infeasible iff A < B/2 (prefix = A - B/2 would be negative).
        for (i, &a) in a_ax.iter().enumerate() {
            for (j, &b) in b_ax.iter().enumerate() {
                assert_eq!(mask[i * b_ax.len() + j], a < b / 2.0, "A={a} B={b}");
            }
        }
        // The low-A / high-B corner is masked, interior is feasible — so both the
        // drop path and the profiled path are exercised.
        assert!(mask.iter().any(|&m| m));
        assert!(mask.iter().any(|&m| !m));
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = config();
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&cfg);
        let payloads = FlashinferAttnPrefillSpec::enumerate(&cfg, &grid, "fa2");

        // 2D cartesian product of the two axes (incl. the infeasible cells, which
        // the build path strips before profiling — enumerate stays full-grid).
        assert_eq!(payloads.len(), grid.axes()[0].len() * grid.axes()[1].len());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, num_qo_heads, num_kv_heads, head_dim, q_dtype,
        // kv_dtype, o_dtype, prefix_len, append_len } — must stay aligned with
        // Python FlashinferAttnPrefillArgs in
        // profiling/kernels/flashinfer_attn_prefill.py.
        assert_eq!(fields.len(), 9);
        assert_eq!(fields.get("backend"), Some(&Value::from("fa2")));
        assert_eq!(fields.get("num_qo_heads"), Some(&Value::from(32_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("o_dtype"), Some(&Value::from("bf16")));
        // First cell is (A=64, B=128): prefix = round(64 - 64) = 0 (fresh prefill).
        assert_eq!(fields.get("prefix_len"), Some(&Value::from(0_u32)));
        assert_eq!(fields.get("append_len"), Some(&Value::from(128_u32)));
        assert_eq!(first.backend(), Some("fa2"));
    }

    fn vllm_fa3_config() -> FlashinferAttnPrefillKernelConfig {
        FlashinferAttnPrefillKernelConfig {
            backends: vec![super::VLLM_FA3_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_qo_heads: 16.into(),
            num_kv_heads: 16.into(),
            head_dim: 256.into(),
            ..config()
        }
    }

    #[test]
    fn vllm_fa3_profiles_only_fresh_prefill_and_keeps_the_combine_edges() {
        // The Python backend rejects prefix_len > 0, so any off-diagonal cell
        // would be a guaranteed profiling failure; and interpolating across
        // 320|321 or 768|769 would blend the split and unsplit FA3 kernels.
        let cfg = vllm_fa3_config();
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&cfg);
        let mask = FlashinferAttnPrefillSpec::infeasible_mask(&cfg, &grid);
        let payloads = FlashinferAttnPrefillSpec::enumerate(&cfg, &grid, "vllm_fa3");
        let feasible: Vec<_> = payloads
            .iter()
            .zip(&mask)
            .filter(|(_, &masked)| !masked)
            .map(|(payload, _)| payload.fields())
            .collect();
        assert_eq!(feasible.len(), 29);
        assert!(feasible
            .iter()
            .all(|fields| fields.get("prefix_len") == Some(&Value::from(0_u32))));
        let appends: Vec<u64> = feasible
            .iter()
            .map(|fields| fields["append_len"].as_u64().unwrap())
            .collect();
        for pair in [[320, 321], [768, 769]] {
            let at = appends.iter().position(|&a| a == pair[0]).unwrap();
            assert_eq!(appends[at + 1], pair[1]);
        }
        // A fresh-prefill query lands on the diagonal: (k=0, q=321) -> (160.5, 321).
        let input = FlashinferAttnPrefillKernelInput {
            prefix_len: 0,
            append_len: 321,
        };
        assert!(grid.axes()[0].contains(&input.coords()[0]));
    }

    #[test]
    fn vllm_fa3_cannot_share_a_config_with_flashinfer_backends() {
        let mixed = FlashinferAttnPrefillKernelConfig {
            backends: vec!["fa3", super::VLLM_FA3_BACKEND],
            ..vllm_fa3_config()
        };
        assert!(FlashinferAttnPrefillSpec::validate_config(&mixed).is_err());
        assert!(FlashinferAttnPrefillSpec::validate_config(&vllm_fa3_config()).is_ok());
        // The FlashInfer grid is untouched.
        assert_eq!(
            FlashinferAttnPrefillSpec::sweep_grid(&config()).axes()[1].len(),
            19
        );
    }
}
