//! GLM-5.3-Flash routed MoE on one TP rank through vLLM's Triton FP8 path
//! (H200, no expert parallelism).
//!
//! Every rank holds all experts with the intermediate dimension sharded by
//! TP, so each rank routes the whole batch and the ranks are symmetric. The
//! launch order follows the H200 capture (vLLM v0.31, `TRITON` Fp8 MoE
//! backend): the BF16 router GEMM with fp32 logits (once per layer, unlike the
//! B200 fork's recompute), `single_group_topk`, the per-token-group FP8 quant
//! of the hidden states, `moe_align_block_size` (+ `count_and_sort`), the
//! `fused_moe_kernel` gate/up launch, SiLU-and-mul, the FP8 quant of the
//! expanded intermediate, the `fused_moe_kernel` down launch, and
//! `moe_sum`.
//!
//! The router GEMM and top-k are separate slots ([`Glm53TritonMoeRouterLocalWorklet`])
//! because decode runs them with the routed path on vLLM's auxiliary stream
//! while the shared expert stays on the main stream; the arch owns that
//! `Parallel` composition.
//!
//! Placeholders: `single_group_topk` (sigmoid `noaux_tc` grouped top-k has no
//! L1 kind; `moe_fused_topk` is the softmax kernel) and SiLU-and-mul are
//! byte-sized elementwise leaves.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, MoeAlignBlockSizeKernel, MoeAlignBlockSizeKernelConfig,
    MoeAlignBlockSizeKernelInput, MoeSumKernel, MoeSumKernelConfig, MoeSumKernelInput,
    VllmFusedMoeDemandKernel, VllmFusedMoeDemandKernelConfig, VllmFusedMoeDemandKernelInput,
    LAUNCH_ROLE_DOWN, LAUNCH_ROLE_GATE_UP,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero};

/// Hopper DeepGEMM / Triton block-FP8 read column-major UE8M0 scales.
pub const COLUMN_MAJOR_SCALE_FORMAT: &str = "ue8m0_column_major";
const QUANT_GROUP_SIZE: u32 = 128;
/// vLLM `get_default_config` for block FP8 (no tuned config exists for
/// E=288, N=512 on H200): `BLOCK_SIZE_M` is 16 up to 64 tokens, else 64.
const SMALL_BATCH_BLOCK_M: u32 = 16;
const LARGE_BATCH_BLOCK_M: u32 = 64;
const SMALL_BATCH_MAX_TOKENS: u32 = 64;

/// Which measured `vllm_fused_moe` rows price the two Triton launches. vLLM
/// reads a tuned config folder (`VLLM_TUNED_CONFIG_FOLDER`) before its shipped
/// tables, so a tuned deployment runs the same kernel with other tile
/// configs; its rows live under their own L1 backend so they never overwrite
/// the default ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusedMoeBackend {
    /// No tuned folder: vLLM's shipped table or `get_default_config`.
    #[default]
    VllmTriton,
    /// `VLLM_TUNED_CONFIG_FOLDER` holding the H200 E=288, N=512 block-FP8
    /// config whose sha256 begins 212d200c (`profiling/configs/vllm_fused_moe/
    /// tuned_212d200c`).
    #[serde(rename = "vllm_triton_tuned_212d200c")]
    VllmTritonTuned212d200c,
}

impl FusedMoeBackend {
    /// The `vllm_fused_moe` L1 backend that prices both launches.
    pub fn l1_backend(self) -> &'static str {
        match self {
            Self::VllmTriton => "vllm_triton",
            Self::VllmTritonTuned212d200c => "vllm_triton_tuned_212d200c",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Glm53TritonMoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53TritonMoeRouterLocalWorkletResolved {
    pub raw_cfg: Glm53TritonMoeRouterLocalWorkletConfig,
    pub gate: GemmFp32OutputKernelConfig,
    pub topk: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53TritonMoeRouterLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53TritonMoeRouterLocalWorklet {
    pub name: String,
    pub gate: Op<GemmFp32OutputKernel>,
    pub topk: Op<ElementwiseKernel>,
}

impl Glm53TritonMoeRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53TritonMoeRouterLocalWorkletConfig,
    ) -> Glm53TritonMoeRouterLocalWorkletResolved {
        let fp32 = DType::Fp32.size_bytes();
        Glm53TritonMoeRouterLocalWorkletResolved {
            gate: GemmFp32OutputKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden.clone(),
                input_dtype: cfg.activation_dtype,
            },
            // Reads the fp32 logits (and bias), writes top-k fp32 weights and
            // int32 ids.
            topk: elementwise(
                &cfg.elementwise_backends,
                &cfg.gpu_name,
                cfg.num_experts.get() * fp32,
                cfg.top_k * 2 * fp32,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53TritonMoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let n = name.as_str();
        Ok(Self {
            gate: atomic(
                n,
                "gate",
                resolved.gate,
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            topk: atomic(n, "topk", resolved.topk, ElementwiseKernel::build, bridge)?,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (Glm53TritonMoeRouterLocalWorklet) [gate once + grouped top-k]",
                self.name
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate.compile(builder),
                self.topk.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53TritonMoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let zero = input.num_tokens == 0;
        push_or_zero(
            &self.gate,
            GemmFp32OutputKernelInput {
                m: input.num_tokens,
            },
            zero,
            ev,
        );
        push_or_zero(
            &self.topk,
            ElementwiseKernelInput {
                num_tokens: input.num_tokens,
            },
            zero,
            ev,
        );
    }
}

#[derive(Clone, Debug)]
pub struct Glm53TritonMoeTpLocalWorkletConfig {
    pub hidden: Dim,
    /// `moe_intermediate_size / tp`: this rank's shard of every expert.
    pub intermediate_per_rank: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub fused_moe_backends: Vec<&'static str>,
    pub align_backends: Vec<&'static str>,
    pub sum_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub expert_demand: ExpertDemand,
}

#[derive(Clone, Debug)]
pub struct Glm53TritonMoeTpLocalWorkletResolved {
    pub raw_cfg: Glm53TritonMoeTpLocalWorkletConfig,
    pub input_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub align: MoeAlignBlockSizeKernelConfig,
    pub gate_up: VllmFusedMoeDemandKernelConfig,
    pub act_and_mul: ElementwiseKernelConfig,
    pub down_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub down: VllmFusedMoeDemandKernelConfig,
    pub sum: MoeSumKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53TritonMoeTpLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53TritonMoeTpLocalWorklet {
    pub name: String,
    pub input_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub align: Op<MoeAlignBlockSizeKernel>,
    pub gate_up: Op<VllmFusedMoeDemandKernel>,
    pub act_and_mul: Op<ElementwiseKernel>,
    pub down_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub down: Op<VllmFusedMoeDemandKernel>,
    pub sum: Op<MoeSumKernel>,
    resolved: Glm53TritonMoeTpLocalWorkletResolved,
}

impl Glm53TritonMoeTpLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53TritonMoeTpLocalWorkletConfig,
    ) -> Glm53TritonMoeTpLocalWorkletResolved {
        assert_eq!(
            cfg.expert_demand.num_experts(),
            cfg.num_experts.get() as usize,
            "demand source width must match num_experts"
        );
        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: QUANT_GROUP_SIZE,
            input_dtype: cfg.activation_dtype,
            scale_format: COLUMN_MAJOR_SCALE_FORMAT.to_string(),
        };
        let fused = |n: Dim, k: Dim, role: &str| VllmFusedMoeDemandKernelConfig {
            backends: cfg.fused_moe_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Fp8E4m3,
            experts_per_token: cfg.top_k,
            launch_role: role.to_string(),
            block_size: QUANT_GROUP_SIZE,
            num_experts: cfg.num_experts.clone(),
            num_local_experts: cfg.num_experts.clone(),
            expert_demand: cfg.expert_demand.clone(),
            folded_rank_position: 0,
        };
        let inter = cfg.intermediate_per_rank.get();
        let bpe = cfg.activation_dtype.size_bytes();
        Glm53TritonMoeTpLocalWorkletResolved {
            input_quant: quant(cfg.hidden.clone()),
            align: MoeAlignBlockSizeKernelConfig {
                backends: cfg.align_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_experts: cfg.num_experts.clone(),
                top_k: cfg.top_k,
            },
            gate_up: fused((2 * inter).into(), cfg.hidden.clone(), LAUNCH_ROLE_GATE_UP),
            // One row per (token, selected expert): reads gate|up, writes act.
            act_and_mul: elementwise(
                &cfg.elementwise_backends,
                &cfg.gpu_name,
                2 * inter * bpe,
                inter * bpe,
            ),
            down_quant: quant(cfg.intermediate_per_rank.clone()),
            down: fused(
                cfg.hidden.clone(),
                cfg.intermediate_per_rank.clone(),
                LAUNCH_ROLE_DOWN,
            ),
            sum: MoeSumKernelConfig {
                backends: cfg.sum_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                top_k: cfg.top_k,
                hidden_dim: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53TritonMoeTpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            input_quant: atomic(
                n,
                "input_quant",
                r.input_quant,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?,
            align: atomic(n, "align", r.align, MoeAlignBlockSizeKernel::build, bridge)?,
            gate_up: atomic(
                n,
                "gate_up",
                r.gate_up,
                VllmFusedMoeDemandKernel::build,
                bridge,
            )?,
            act_and_mul: atomic(
                n,
                "act_and_mul",
                r.act_and_mul,
                ElementwiseKernel::build,
                bridge,
            )?,
            down_quant: atomic(
                n,
                "down_quant",
                r.down_quant,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, VllmFusedMoeDemandKernel::build, bridge)?,
            sum: atomic(n, "sum", r.sum, MoeSumKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53TritonMoeTpLocalWorklet) [TP rank; all {} experts, top-{}, I/rank={}]",
                self.name, cfg.num_experts, cfg.top_k, cfg.intermediate_per_rank
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_quant.compile(builder),
                self.align.compile(builder),
                self.gate_up.compile(builder),
                self.act_and_mul.compile(builder),
                self.down_quant.compile(builder),
                self.down.compile(builder),
                self.sum.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53TritonMoeTpLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let tokens = input.num_tokens;
        let zero = tokens == 0;
        let selections = tokens
            .checked_mul(cfg.top_k)
            .expect("tokens x top_k must fit u32");
        push_or_zero(
            &self.input_quant,
            Fp8PerTokenGroupQuantKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        push_or_zero(
            &self.align,
            MoeAlignBlockSizeKernelInput {
                num_tokens: tokens,
                block_size: block_size_m(tokens),
            },
            zero || naive_block_assignment(tokens, cfg.top_k, cfg.num_experts.get()),
            ev,
        );
        push_or_zero(
            &self.gate_up,
            VllmFusedMoeDemandKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        let expanded = ElementwiseKernelInput {
            num_tokens: selections,
        };
        push_or_zero(&self.act_and_mul, expanded, zero, ev);
        push_or_zero(
            &self.down_quant,
            Fp8PerTokenGroupQuantKernelInput {
                num_tokens: selections,
            },
            zero,
            ev,
        );
        push_or_zero(
            &self.down,
            VllmFusedMoeDemandKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        push_or_zero(
            &self.sum,
            MoeSumKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
    }
}

/// vLLM's Triton `BLOCK_SIZE_M` for this token count (default config).
fn block_size_m(tokens: u32) -> u32 {
    if tokens <= SMALL_BATCH_MAX_TOKENS {
        SMALL_BATCH_BLOCK_M
    } else {
        LARGE_BATCH_BLOCK_M
    }
}

/// vLLM skips `moe_align_block_size` when `tokens * top_k * 4 <= experts`
/// (`_prepare_expert_assignment`'s sparsity heuristic, no expert map).
fn naive_block_assignment(tokens: u32, top_k: u32, num_experts: u32) -> bool {
    u64::from(tokens) * u64::from(top_k) * 4 <= u64::from(num_experts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::routing::RoutingDistribution;

    fn cfg() -> Glm53TritonMoeTpLocalWorkletConfig {
        Glm53TritonMoeTpLocalWorkletConfig {
            hidden: 4096.into(),
            intermediate_per_rank: 512.into(),
            num_experts: 288.into(),
            top_k: 8,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            quant_backends: vec!["vllm_cuda"],
            fused_moe_backends: vec!["vllm_triton"],
            align_backends: vec!["vllm_cuda"],
            sum_backends: vec!["vllm_cuda"],
            elementwise_backends: vec!["triton"],
            expert_demand: ExpertDemand::popularity(&RoutingDistribution::uniform(288), 1),
        }
    }

    #[test]
    fn every_rank_holds_every_expert_with_a_tp_shard_of_the_intermediate() {
        let r = Glm53TritonMoeTpLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.gate_up.num_local_experts.get(), 288);
        assert_eq!(r.gate_up.n.get(), 1024);
        assert_eq!(r.gate_up.k.get(), 4096);
        assert_eq!(r.down.n.get(), 4096);
        assert_eq!(r.down.k.get(), 512);
        assert_eq!(r.gate_up.launch_role, LAUNCH_ROLE_GATE_UP);
        assert_eq!(r.down.launch_role, LAUNCH_ROLE_DOWN);
        assert_eq!(r.input_quant.scale_format, COLUMN_MAJOR_SCALE_FORMAT);
        assert_eq!(r.down_quant.hidden_size.get(), 512);
    }

    #[test]
    fn align_follows_vllm_naive_assignment_and_block_m() {
        // 288 experts, top-8: tokens * 32 <= 288 up to 9 tokens.
        assert!(naive_block_assignment(9, 8, 288));
        assert!(!naive_block_assignment(10, 8, 288));
        assert_eq!(block_size_m(64), 16);
        assert_eq!(block_size_m(65), 64);
    }
}
