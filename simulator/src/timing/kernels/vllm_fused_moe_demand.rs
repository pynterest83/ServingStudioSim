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

    /// A 6-wide verify corpus over the 288-expert, top-8 shape: the H200
    /// GLM-5.3-Flash MTP-5 binding in miniature.
    fn verify_corpus(tag: &str) -> ExpertDemand {
        let dir = crate::timing::token_corpus::tests::temp_dir(tag);
        crate::timing::token_corpus::tests::synthetic(&dir, 288, 8, 2, 64);
        let manifest = dir.join("manifest.json");
        ExpertDemand::corpus(manifest.to_str().unwrap(), 6, 0..2).unwrap()
    }

    fn token_axis(expert_demand: ExpertDemand) -> Vec<u32> {
        let config = VllmFusedMoeDemandKernelConfig {
            expert_demand,
            ..config()
        };
        let axis = VllmFusedMoeDemandSpec::sweep_grid(&config).axes()[0].clone();
        axis.into_iter().map(|tokens| tokens as u32).collect()
    }

    /// Without a bound the grid is the one every cached corpus config was
    /// profiled on: the fixed head, the shared token axis, then 6 x (1..=8).
    #[test]
    fn a_corpus_without_a_bound_keeps_its_profiled_grid() {
        let head = [1, 4, 8, 16, 32, 48];
        let mut expected = head.to_vec();
        expected.extend(
            Axis::token_axis()
                .into_iter()
                .map(|tokens| tokens as u32)
                .filter(|tokens| !head.contains(tokens)),
        );
        expected.extend([6, 12, 18, 24, 30, 36, 42]);
        assert_eq!(token_axis(verify_corpus("axis-default")), expected);
    }

    /// A bound of 144 adds exactly the verify multiples 54..=144 -- the
    /// 66-102-row decode batches the default grid crossed with one segment.
    #[test]
    fn a_bound_adds_the_verify_multiples_up_to_it() {
        let default = token_axis(verify_corpus("axis-bound-base"));
        let bounded = token_axis(
            verify_corpus("axis-bound")
                .with_axis_max_tokens(144)
                .unwrap(),
        );
        assert_eq!(bounded[..default.len()], default[..]);
        let added: Vec<u32> = bounded[default.len()..].to_vec();
        assert_eq!(added, (9..=24).map(|blocks| 6 * blocks).collect::<Vec<_>>());
        // A bound between multiples stops at the last whole block.
        let floored = token_axis(
            verify_corpus("axis-bound-floor")
                .with_axis_max_tokens(149)
                .unwrap(),
        );
        assert_eq!(floored, bounded);
    }

    /// The bound reaches the kernel-config record only when it is set, so a
    /// config built without one keeps the identity its rows were recorded under.
    #[test]
    fn the_identity_names_the_bound_only_when_set() {
        let demand = verify_corpus("axis-identity");
        let identity = |expert_demand: ExpertDemand| {
            VllmFusedMoeDemandKernelConfig {
                expert_demand,
                ..config()
            }
            .identity()
        };
        let default = identity(demand.clone());
        let corpus = &default["expert_demand"]["corpus"];
        assert!(corpus.is_object(), "{default}");
        assert!(corpus.get("axis_max_tokens").is_none(), "{default}");
        let mut keys: Vec<&str> = corpus
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "checksum_fnv1a64",
                "data_file",
                "group_size",
                "layer_end",
                "layer_start",
                "num_experts",
                "num_layers",
                "num_tokens",
                "sampling_candidates",
                "schema_version",
                "seed",
                "top_k",
            ]
        );

        let bounded = identity(demand.with_axis_max_tokens(144).unwrap());
        assert_eq!(bounded["expert_demand"]["corpus"]["axis_max_tokens"], 144);
        let mut stripped = bounded.clone();
        stripped["expert_demand"]["corpus"]
            .as_object_mut()
            .unwrap()
            .remove("axis_max_tokens");
        assert_eq!(stripped, default, "the bound is the only identity change");
        // A config serialized without the field reads back as no bound.
        let ExpertDemand::Corpus(reread) = serde_json::from_value::<ExpertDemand>(
            serde_json::to_value(verify_corpus("axis-reread")).unwrap(),
        )
        .unwrap() else {
            panic!("a corpus reads back as a corpus");
        };
        assert_eq!(reread.axis_max_tokens, 0);
    }
}
