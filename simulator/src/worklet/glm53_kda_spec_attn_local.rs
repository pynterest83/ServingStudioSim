//! GLM-5.3-Flash Kimi Delta Attention on one TP rank under MTP speculative
//! decoding (vLLM v0.31 `glm5next/common/kda.py`, H200).
//!
//! Unlike the B200 MTP-off path (`glm53_kda_attn_local`), the verify rows and
//! the prefill rows take different cores in the same iteration:
//!
//! - spec-verify rows (K+1 tokens per request): `causal_conv1d_update` with
//!   `query_start_loc`, then `fused_recurrent_kda` with `num_accepted_tokens`;
//! - prefill rows: `causal_conv1d_fn`, `gather_initial_states`, the FlashKDA
//!   prefill (`kda_prefill_backend` auto on Hopper), `scatter_states`.
//!
//! A mixed iteration additionally splits the projections into the two token
//! groups (`index_select` x6: qkv, g, beta for each group) and scatters the
//! two core outputs back (two index copies); a pure verify step skips both.
//!
//! First-cut approximations, labeled until their L1 lands (plan M4):
//! - the verify conv and recurrence cost a `batch = requests x (K+1)` launch of
//!   the single-token kernels (no query-length axis yet);
//! - the prefill core uses `kda_chunk_prefill` (`chunk_kda`), not FlashKDA;
//! - split/merge/copy glue and the state gather/scatter are byte-sized
//!   elementwise leaves, the gated RMSNorm too (as on B200).

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GdnCausalConvDecodeKernel,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernel,
    GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput, KdaChunkPrefillKernel,
    KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput, KdaRecurrentDecodeKernel,
    KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

const STATE_TRANSFER_UNIT_BYTES: u32 = 4096;
/// Mixed steps: `index_select` of qkv, g and beta into each token group.
const SPLIT_SELECT_LAUNCHES: u32 = 6;
/// Prefill-bearing contiguous copies ahead of the FlashKDA prefill.
const PREFILL_COPY_LAUNCHES: u32 = 4;
/// Mixed steps: the two group outputs scattered back into one buffer.
const MERGE_COPY_LAUNCHES: u32 = 2;

#[derive(Clone, Debug)]
pub struct Glm53KdaSpecAttnLocalWorkletConfig {
    pub hidden: Dim,
    /// KDA heads on this rank.
    pub num_heads: Dim,
    pub head_dim: Dim,
    /// Width of the `f_a` / `g_a` low-rank gate factors.
    pub gate_rank: u32,
    pub conv_kernel_size: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub conv_backends: Vec<&'static str>,
    pub prefill_core_backends: Vec<&'static str>,
    pub decode_core_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53KdaSpecAttnLocalWorkletResolved {
    pub raw_cfg: Glm53KdaSpecAttnLocalWorkletConfig,
    pub in_proj: SingleGemmKernelConfig,
    pub f_b: SingleGemmKernelConfig,
    pub g_b: SingleGemmKernelConfig,
    pub split_select: ElementwiseKernelConfig,
    pub conv_verify: GdnCausalConvDecodeKernelConfig,
    pub conv_prefill: GdnCausalConvPrefillKernelConfig,
    pub recurrent_verify: KdaRecurrentDecodeKernelConfig,
    pub state_gather: ElementwiseKernelConfig,
    pub prefill_copy: ElementwiseKernelConfig,
    pub chunk_prefill: KdaChunkPrefillKernelConfig,
    pub state_scatter: ElementwiseKernelConfig,
    pub merge_copy: ElementwiseKernelConfig,
    pub gated_norm: ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    /// fp32 `[H, K, V]` recurrent state of one request on this rank.
    pub ssm_state_bytes_per_request: u32,
    /// bf16 `(kernel_size - 1)`-token conv window of one request on this rank.
    pub conv_state_bytes_per_request: u32,
}

/// One iteration's KDA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53KdaSpecAttnLocalWorkletInput {
    /// Appended tokens of each prefill request.
    pub prefill_sequence_lengths: Vec<u32>,
    /// Requests in speculative verify (each contributes `verify_width` rows).
    pub verify_requests: u32,
    /// K+1 rows per verifying request.
    pub verify_width: u32,
}

pub struct Glm53KdaSpecAttnLocalWorklet {
    pub name: String,
    pub in_proj: Op<SingleGemmKernel>,
    pub f_b: Op<SingleGemmKernel>,
    pub g_b: Op<SingleGemmKernel>,
    pub split_select: Op<ElementwiseKernel>,
    pub conv_verify: Op<GdnCausalConvDecodeKernel>,
    pub conv_prefill: Op<GdnCausalConvPrefillKernel>,
    pub recurrent_verify: Op<KdaRecurrentDecodeKernel>,
    pub state_gather: Op<ElementwiseKernel>,
    pub prefill_copy: Op<ElementwiseKernel>,
    pub chunk_prefill: Op<KdaChunkPrefillKernel>,
    pub state_scatter: Op<ElementwiseKernel>,
    pub merge_copy: Op<ElementwiseKernel>,
    pub gated_norm: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: Glm53KdaSpecAttnLocalWorkletResolved,
}

impl Glm53KdaSpecAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53KdaSpecAttnLocalWorkletConfig,
    ) -> Glm53KdaSpecAttnLocalWorkletResolved {
        let heads = cfg.num_heads.get();
        let head_dim = cfg.head_dim.get();
        let value_width = heads * head_dim;
        let bpe = cfg.activation_dtype.size_bytes();
        // q, k, v, one beta per head, then the f_a and g_a gate factors.
        let in_proj_width = 3 * value_width + heads + 2 * cfg.gate_rank;
        let conv_channels = 3 * value_width;
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.bf16_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.activation_dtype,
        };
        let ew = |input: u32, output: u32| {
            elementwise(&cfg.elementwise_backends, &cfg.gpu_name, input, output)
        };
        let per_head_bytes = value_width * bpe;
        let ssm_state_bytes_per_request = heads * head_dim * head_dim * DType::Fp32.size_bytes();
        assert_eq!(
            ssm_state_bytes_per_request % STATE_TRANSFER_UNIT_BYTES,
            0,
            "KDA state must be a whole number of transfer units"
        );
        let conv = |backends: &Vec<&'static str>| GdnCausalConvDecodeKernelConfig {
            backends: backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            channels: conv_channels.into(),
            kernel_size: cfg.conv_kernel_size.clone(),
            dtype: cfg.activation_dtype,
            state_dtype: cfg.activation_dtype,
        };
        Glm53KdaSpecAttnLocalWorkletResolved {
            in_proj: gemm(in_proj_width.into(), cfg.hidden.clone()),
            f_b: gemm(value_width.into(), cfg.gate_rank.into()),
            g_b: gemm(value_width.into(), cfg.gate_rank.into()),
            // Largest of the selected tensors: one q/k/v-sized row.
            split_select: ew(per_head_bytes, per_head_bytes),
            conv_verify: conv(&cfg.conv_backends),
            conv_prefill: GdnCausalConvPrefillKernelConfig {
                backends: cfg.conv_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: conv_channels.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.activation_dtype,
            },
            recurrent_verify: KdaRecurrentDecodeKernelConfig {
                backends: cfg.decode_core_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            state_gather: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            prefill_copy: ew(per_head_bytes, per_head_bytes),
            chunk_prefill: KdaChunkPrefillKernelConfig {
                backends: cfg.prefill_core_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            state_scatter: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            merge_copy: ew(per_head_bytes, per_head_bytes),
            // rmsnorm(o) * w * sigmoid(g): reads o and g, writes the gated o.
            gated_norm: ew(2 * per_head_bytes, per_head_bytes),
            o_proj: gemm(cfg.hidden.clone(), value_width.into()),
            ssm_state_bytes_per_request,
            conv_state_bytes_per_request: conv_channels * (cfg.conv_kernel_size.get() - 1) * bpe,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53KdaSpecAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            in_proj: atomic(n, "in_proj", r.in_proj, SingleGemmKernel::build, bridge)?,
            f_b: atomic(n, "f_b_proj", r.f_b, SingleGemmKernel::build, bridge)?,
            g_b: atomic(n, "g_b_proj", r.g_b, SingleGemmKernel::build, bridge)?,
            split_select: atomic(
                n,
                "split_select",
                r.split_select,
                ElementwiseKernel::build,
                bridge,
            )?,
            conv_verify: atomic(
                n,
                "short_conv_verify",
                r.conv_verify,
                GdnCausalConvDecodeKernel::build,
                bridge,
            )?,
            conv_prefill: atomic(
                n,
                "short_conv_prefill",
                r.conv_prefill,
                GdnCausalConvPrefillKernel::build,
                bridge,
            )?,
            recurrent_verify: atomic(
                n,
                "recurrent_verify",
                r.recurrent_verify,
                KdaRecurrentDecodeKernel::build,
                bridge,
            )?,
            state_gather: atomic(
                n,
                "state_gather",
                r.state_gather,
                ElementwiseKernel::build,
                bridge,
            )?,
            prefill_copy: atomic(
                n,
                "prefill_copy",
                r.prefill_copy,
                ElementwiseKernel::build,
                bridge,
            )?,
            chunk_prefill: atomic(
                n,
                "chunk_prefill",
                r.chunk_prefill,
                KdaChunkPrefillKernel::build,
                bridge,
            )?,
            state_scatter: atomic(
                n,
                "state_scatter",
                r.state_scatter,
                ElementwiseKernel::build,
                bridge,
            )?,
            merge_copy: atomic(
                n,
                "merge_copy",
                r.merge_copy,
                ElementwiseKernel::build,
                bridge,
            )?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                ElementwiseKernel::build,
                bridge,
            )?,
            o_proj: atomic(n, "o_proj", r.o_proj, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn resolved(&self) -> &Glm53KdaSpecAttnLocalWorkletResolved {
        &self.resolved
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53KdaSpecAttnLocalWorklet) [TP rank; H={}, D={}; verify rows recurrent, prefill rows chunked]",
                self.name, cfg.num_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(vec![
                self.in_proj.compile(builder),
                self.f_b.compile(builder),
                self.g_b.compile(builder),
                repeated(&self.split_select, SPLIT_SELECT_LAUNCHES, builder),
                self.conv_verify.compile(builder),
                self.conv_prefill.compile(builder),
                self.recurrent_verify.compile(builder),
                self.state_gather.compile(builder),
                repeated(&self.prefill_copy, PREFILL_COPY_LAUNCHES, builder),
                self.chunk_prefill.compile(builder),
                self.state_scatter.compile(builder),
                repeated(&self.merge_copy, MERGE_COPY_LAUNCHES, builder),
                self.gated_norm.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53KdaSpecAttnLocalWorkletInput, ev: &mut Evaluator) {
        let w = derive_work(input, self.resolved.ssm_state_bytes_per_request)
            .unwrap_or_else(|reason| panic!("invalid Glm53KdaSpecAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let all_tokens = ElementwiseKernelInput {
            num_tokens: w.total_tokens,
        };
        let no_prefill = w.prefill_tokens == 0;
        let no_verify = w.verify_rows == 0;
        let mixed = !no_prefill && !no_verify;
        push_or_zero(&self.in_proj, rows.clone(), false, ev);
        push_or_zero(&self.f_b, rows.clone(), false, ev);
        push_or_zero(&self.g_b, rows.clone(), false, ev);
        push_or_zero(&self.split_select, all_tokens.clone(), !mixed, ev);
        // Approximation: K+1 sequential rows per request as K+1 one-row requests.
        push_or_zero(
            &self.conv_verify,
            GdnCausalConvDecodeKernelInput {
                batch_size: w.verify_rows,
            },
            no_verify,
            ev,
        );
        push_or_zero(
            &self.conv_prefill,
            GdnCausalConvPrefillKernelInput {
                batch_size: 1,
                sequence_length: w.prefill_tokens,
            },
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.recurrent_verify,
            KdaRecurrentDecodeKernelInput {
                batch_size: w.verify_rows,
            },
            no_verify,
            ev,
        );
        let state = ElementwiseKernelInput {
            num_tokens: w.prefill_state_units,
        };
        push_or_zero(&self.state_gather, state.clone(), no_prefill, ev);
        push_or_zero(
            &self.prefill_copy,
            ElementwiseKernelInput {
                num_tokens: w.prefill_tokens,
            },
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.chunk_prefill,
            KdaChunkPrefillKernelInput {
                num_tokens: w.prefill_tokens,
                max_sequence_length: w.max_prefill_length,
                num_decode_sequences: 0,
            },
            no_prefill,
            ev,
        );
        push_or_zero(&self.state_scatter, state, no_prefill, ev);
        push_or_zero(&self.merge_copy, all_tokens.clone(), !mixed, ev);
        push_or_zero(&self.gated_norm, all_tokens, false, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

struct Work {
    total_tokens: u32,
    prefill_tokens: u32,
    verify_rows: u32,
    max_prefill_length: u32,
    /// 4-KiB units of every prefill request's state.
    prefill_state_units: u32,
}

fn derive_work(
    input: &Glm53KdaSpecAttnLocalWorkletInput,
    state_bytes: u32,
) -> Result<Work, String> {
    let mut prefill_tokens = 0_u32;
    let mut max_prefill_length = 0_u32;
    for (index, &length) in input.prefill_sequence_lengths.iter().enumerate() {
        if length == 0 {
            return Err(format!("prefill sequence {index} must be positive"));
        }
        prefill_tokens = prefill_tokens
            .checked_add(length)
            .ok_or("prefill token sum overflows u32")?;
        max_prefill_length = max_prefill_length.max(length);
    }
    if input.verify_requests > 0 && input.verify_width == 0 {
        return Err("verify_width must be positive when requests verify".into());
    }
    let verify_rows = input
        .verify_requests
        .checked_mul(input.verify_width)
        .ok_or("verify rows overflow u32")?;
    let total_tokens = prefill_tokens
        .checked_add(verify_rows)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let prefill_state_units = (input.prefill_sequence_lengths.len() as u32)
        .checked_mul(state_bytes / STATE_TRANSFER_UNIT_BYTES)
        .ok_or("state transfer units overflow u32")?;
    Ok(Work {
        total_tokens,
        prefill_tokens,
        verify_rows,
        max_prefill_length,
        prefill_state_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Glm53KdaSpecAttnLocalWorkletConfig {
        Glm53KdaSpecAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_heads: 16.into(),
            head_dim: 128.into(),
            gate_rank: 128,
            conv_kernel_size: 4.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            bf16_gemm_backends: vec!["torch_linear_vllm"],
            conv_backends: vec!["vllm_triton"],
            prefill_core_backends: vec!["vllm_triton"],
            decode_core_backends: vec!["vllm_triton"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn verify_and_prefill_rows_take_separate_cores() {
        let w = derive_work(
            &Glm53KdaSpecAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![2048, 256],
                verify_requests: 15,
                verify_width: 6,
            },
            1 << 20,
        )
        .unwrap();
        assert_eq!((w.prefill_tokens, w.verify_rows), (2304, 90));
        assert_eq!(w.total_tokens, 2394);
        assert_eq!(w.max_prefill_length, 2048);
        // Only prefill requests gather and scatter state.
        assert_eq!(w.prefill_state_units, 2 * 256);
        assert!(derive_work(
            &Glm53KdaSpecAttnLocalWorkletInput {
                prefill_sequence_lengths: Vec::new(),
                verify_requests: 3,
                verify_width: 0,
            },
            1 << 20
        )
        .is_err());
    }

    #[test]
    fn compile_has_fixed_slots_for_both_paths() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = Glm53KdaSpecAttnLocalWorklet::build(
            "m.kda".into(),
            Glm53KdaSpecAttnLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let names: Vec<_> = tree.slots.iter().map(|slot| slot.name.as_str()).collect();
        assert_eq!(names.len(), 14);
        assert!(names.contains(&"m.kda.recurrent_verify"));
        assert!(names.contains(&"m.kda.chunk_prefill"));
    }
}
