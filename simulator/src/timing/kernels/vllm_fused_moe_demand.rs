//! vLLM Triton `fused_moe_kernel` driven by an [`ExpertDemand`] source.
//!
//! The same physical launch and profile.db table as [`super::vllm_fused_moe`]
//! (`profile_kind` is `vllm_fused_moe`, the Python args are identical), but
//! the per-expert row histogram comes from a routing source instead of a fixed
//! ppm vector: a measured token corpus keeps the temporal structure of real
//! routing (a speculative verify block routes its K+1 positions almost alike),
//! which a marginal cannot. A config never changes the other kernel's identity,
//! so archs on `local_ppm` keep their caches.
//!
//! The kernel serves a TP-only deployment where every rank holds every expert:
//! `num_local_experts == num_experts` and the histogram is the global one.
//! `folded_rank_position` stays explicit so an EP split can reuse the leaf.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VllmFusedMoeDemandKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub experts_per_token: u32,
    /// `"w13"` or `"w2"`, as in `vllm_fused_moe`.
    pub launch_role: String,
    /// FP8 activation/weight scale granularity.
    pub block_size: u32,
    pub num_experts: Dim,
    pub num_local_experts: Dim,
    pub expert_demand: ExpertDemand,
    pub folded_rank_position: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct VllmFusedMoeDemandKernelInput {
    pub num_tokens: u32,
}

pub struct VllmFusedMoeDemandSpec;

impl KernelSpec for VllmFusedMoeDemandSpec {
    type Config = VllmFusedMoeDemandKernelConfig;
    type Input = VllmFusedMoeDemandKernelInput;

    const KIND: KernelKind = "vllm_fused_moe_demand";

    fn profile_kind() -> KernelKind {
        "vllm_fused_moe"
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        // The `vllm_fused_moe` token axis, refined by the demand source
        // (a corpus samples whole verify blocks).
        SweepGrid::new(vec![config.expert_demand.token_axis(Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ]))])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        config.expert_demand.prepare().map(|_| ())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let token_counts = grid.expand_1d(|num_tokens| num_tokens as u32);
        let mut batches = config
            .expert_demand
            .per_expert_batches(
                config.experts_per_token,
                &token_counts,
                config.num_experts.get() as usize,
                config.num_local_experts.get() as usize,
                config.folded_rank_position,
            )
            .expect("validate_config proved this source readable")
            .into_iter();

        grid.expand_1d(|num_tokens| {
            let per_group_batches = batches.next().expect("one histogram per grid point");
            ArgsPayload::new()
                .with("backend", backend)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
                .with("num_local_experts", config.num_local_experts.get())
                .with("num_tokens", num_tokens as u32)
                .with("experts_per_token", config.experts_per_token)
                .with("launch_role", config.launch_role.as_str())
                .with("block_size", config.block_size)
                .with("per_group_batches", per_group_batches)
        })
    }
}

register_kernel!(VllmFusedMoeDemandKernel, VllmFusedMoeDemandSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::vllm_fused_moe::{
        VllmFusedMoeKernelConfig, VllmFusedMoeSpec, LAUNCH_ROLE_GATE_UP,
    };
    use crate::timing::routing::RoutingDistribution;

    fn config() -> VllmFusedMoeDemandKernelConfig {
        VllmFusedMoeDemandKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 1024.into(),
            k: 4096.into(),
            dtype: DType::Fp8E4m3,
            experts_per_token: 8,
            launch_role: LAUNCH_ROLE_GATE_UP.to_string(),
            block_size: 128,
            num_experts: 288.into(),
            num_local_experts: 288.into(),
            expert_demand: ExpertDemand::popularity(&RoutingDistribution::uniform(288), 1),
            folded_rank_position: 0,
        }
    }

    #[test]
    fn reads_the_vllm_fused_moe_table() {
        assert_eq!(VllmFusedMoeDemandSpec::KIND, "vllm_fused_moe_demand");
        assert_eq!(VllmFusedMoeDemandSpec::profile_kind(), "vllm_fused_moe");
    }

    /// Same row schema as the ppm leaf: every field the Python ABI reads, and
    /// one histogram entry per local expert summing to tokens x top-k.
    #[test]
    fn args_match_the_vllm_fused_moe_schema() {
        let grid = SweepGrid::new(vec![Axis::values([16])]);
        let payload = &VllmFusedMoeDemandSpec::enumerate(&config(), &grid, "vllm_triton")[0];
        let fields = payload.fields();
        let base = VllmFusedMoeKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 1024.into(),
            k: 4096.into(),
            dtype: DType::Fp8E4m3,
            experts_per_token: 8,
            launch_role: LAUNCH_ROLE_GATE_UP.to_string(),
            block_size: 128,
            local_ppm: vec![1_000_000 / 288; 288],
        };
        let base_payload = &VllmFusedMoeSpec::enumerate(&base, &grid, "vllm_triton")[0];
        let mut keys: Vec<_> = fields.keys().collect();
        let mut base_keys: Vec<_> = base_payload.fields().keys().collect();
        keys.sort();
        base_keys.sort();
        assert_eq!(keys, base_keys);
        let batches = fields["per_group_batches"].as_array().unwrap();
        assert_eq!(batches.len(), 288);
        let rows: u64 = batches.iter().map(|value| value.as_u64().unwrap()).sum();
        assert_eq!(rows, 16 * 8);
        assert!(!fields.contains_key("folded_rank_position"));
    }
}
