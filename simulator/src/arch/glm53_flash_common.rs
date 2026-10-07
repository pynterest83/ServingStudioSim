//! Pieces shared by the GLM-5.3-Flash vLLM archs (the B200 TP=EP arch and the
//! H200 TP-only MTP arch): the checkpoint config, the layer grouping, the mHC
//! sublayer boundary, CUDA-graph padding, and op construction helpers. Moved
//! verbatim from `glm53_flash_vllm_fp8_kda_dsa_moe.rs`; no cost changes.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use crate::op::Op;
use crate::timing::kernels::{
    MhcFusedPostPreRmsNormKernel, MhcPreRmsNormKernel, MhcRmsNormKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe, SlotInput,
};

/// Config-error prefix shared by every GLM-5.3-Flash arch.
const MODEL_KIND: &str = "GLM-5.3-Flash arch";

/// The checkpoint's dimensions, read strictly from its `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm53FlashModelCfg {
    pub hidden: u32,
    pub num_layers: u32,
    pub kda_layers: Vec<u32>,
    pub first_k_dense_replace: u32,
    pub kda_num_heads: u32,
    pub kda_head_dim: u32,
    pub short_conv_kernel_size: u32,
    pub num_attention_heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
    pub intermediate_size: u32,
    pub moe_intermediate_size: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub n_shared_experts: u32,
    pub n_group: u32,
    pub topk_group: u32,
    /// `routed_scaling_factor` as an exact fraction.
    pub routed_scaling: (u32, u32),
    pub hc_mult: u32,
    pub rms_norm_eps: f64,
    pub vocab_size: u32,
}

impl Glm53FlashModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading GLM-5.3-Flash config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing GLM-5.3-Flash config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let cfg = root.get("text_config").unwrap_or(root);
        let u = |key: &str| -> Result<u32> {
            let value = cfg.get(key).and_then(Value::as_u64).with_context(|| {
                format!("GLM-5.3-Flash config field `{key}` must be an integer")
            })?;
            u32::try_from(value).with_context(|| format!("`{key}` exceeds u32"))
        };
        let f = |key: &str| -> Result<f64> {
            cfg.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("GLM-5.3-Flash config field `{key}` must be a number"))
        };
        let model_type = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            model_type == "glm5_next_text" || model_type == "glm5_next",
            "{MODEL_KIND} requires a GLM-5.3-Flash (glm5_next) config, got model_type {model_type:?}"
        );
        let linear = cfg
            .get("linear_attn_config")
            .context("GLM-5.3-Flash config needs linear_attn_config")?;
        let lu = |key: &str| -> Result<u32> {
            let value = linear
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("linear_attn_config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("linear_attn_config.{key} exceeds u32"))
        };
        let kda_layers: Vec<u32> = linear
            .get("kda_layers")
            .and_then(Value::as_array)
            .context("linear_attn_config.kda_layers must be an array")?
            .iter()
            .map(|layer| {
                layer
                    .as_u64()
                    .and_then(|layer| u32::try_from(layer).ok())
                    .context("kda_layers entries must be u32")
            })
            .collect::<Result<_>>()?;
        let routed_scaling_factor = f("routed_scaling_factor")?;
        let doubled = routed_scaling_factor * 2.0;
        ensure!(
            doubled.fract() == 0.0 && doubled > 0.0,
            "routed_scaling_factor {routed_scaling_factor} must be a positive multiple of 1/2"
        );
        let model = Self {
            hidden: u("hidden_size")?,
            num_layers: u("num_hidden_layers")?,
            kda_layers,
            first_k_dense_replace: u("first_k_dense_replace")?,
            kda_num_heads: lu("num_heads")?,
            kda_head_dim: lu("head_dim")?,
            short_conv_kernel_size: lu("short_conv_kernel_size")?,
            num_attention_heads: u("num_attention_heads")?,
            q_lora_rank: u("q_lora_rank")?,
            kv_lora_rank: u("kv_lora_rank")?,
            qk_nope_head_dim: u("qk_nope_head_dim")?,
            v_head_dim: u("v_head_dim")?,
            index_n_heads: u("index_n_heads")?,
            index_head_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            index_kpool: u("index_kpool")?,
            intermediate_size: u("intermediate_size")?,
            moe_intermediate_size: u("moe_intermediate_size")?,
            n_routed_experts: u("n_routed_experts")?,
            num_experts_per_tok: u("num_experts_per_tok")?,
            n_shared_experts: u("n_shared_experts")?,
            n_group: u("n_group")?,
            topk_group: u("topk_group")?,
            routed_scaling: (doubled as u32, 2),
            hc_mult: u("hc_mult")?,
            rms_norm_eps: f("rms_norm_eps")?,
            vocab_size: u("vocab_size")?,
        };
        ensure!(
            u("qk_rope_head_dim")? == 0,
            "{MODEL_KIND} models the no-rope MLA latent; qk_rope_head_dim must be 0"
        );
        ensure!(
            model.n_shared_experts == 1,
            "exactly one shared expert is modeled"
        );
        ensure!(
            model
                .kda_layers
                .iter()
                .all(|&layer| layer < model.num_layers),
            "kda_layers must index model layers"
        );
        ensure!(
            model.first_k_dense_replace <= model.num_layers,
            "first_k_dense_replace exceeds num_hidden_layers"
        );
        Ok(model)
    }

    pub(crate) fn is_kda(&self, layer: u32) -> bool {
        self.kda_layers.contains(&layer)
    }

    pub(crate) fn is_dense(&self, layer: u32) -> bool {
        layer < self.first_k_dense_replace
    }

    pub fn num_dsa_layers(&self) -> u32 {
        (0..self.num_layers).filter(|&l| !self.is_kda(l)).count() as u32
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.num_layers - self.first_k_dense_replace
    }
}

/// Which attention and FFN a layer group carries, and how it opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttnKind {
    Kda,
    Dsa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FfnKind {
    Dense,
    Moe,
}

/// A run of layers with one identical sublayer shape, folded by `Scale{n}`.
#[derive(Clone, Debug)]
pub struct Glm53FlashLayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    pub(crate) attn: AttnKind,
    pub(crate) ffn: FfnKind,
    /// Only layer 0 opens with a standalone mHC pre.
    pub(crate) opens_stream: bool,
}

/// The ordered layer groups of a model config. Layers with the same attention
/// kind, FFN kind, and opening boundary fold into one group.
pub(crate) fn layer_groups(model: &Glm53FlashModelCfg) -> Vec<Glm53FlashLayerGroup> {
    let mut groups: Vec<Glm53FlashLayerGroup> = Vec::new();
    for layer in 0..model.num_layers {
        let attn = if model.is_kda(layer) {
            AttnKind::Kda
        } else {
            AttnKind::Dsa
        };
        let ffn = if model.is_dense(layer) {
            FfnKind::Dense
        } else {
            FfnKind::Moe
        };
        let opens_stream = layer == 0;
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.attn == attn && g.ffn == ffn && g.opens_stream == opens_stream)
        {
            group.layers.push(layer);
            continue;
        }
        let label = format!(
            "{}{}_{}",
            if opens_stream { "first_" } else { "" },
            match attn {
                AttnKind::Kda => "kda",
                AttnKind::Dsa => "dsa",
            },
            match ffn {
                FfnKind::Dense => "dense",
                FfnKind::Moe => "moe",
            }
        );
        groups.push(Glm53FlashLayerGroup {
            label,
            layers: vec![layer],
            attn,
            ffn,
            opens_stream,
        });
    }
    groups
}

pub(crate) enum Boundary {
    Pre(Op<MhcPreRmsNormKernel>),
    Fused(Op<MhcFusedPostPreRmsNormKernel>),
}

impl Boundary {
    pub(crate) fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Pre(op) => op.compile(builder),
            Self::Fused(op) => op.compile(builder),
        }
    }

    pub(crate) fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let input = MhcRmsNormKernelInput { num_tokens };
        match self {
            Self::Pre(op) => push(op, input, ev),
            Self::Fused(op) => push(op, input, ev),
        }
    }
}

/// The row count vLLM runs a CUDA-graph replay on: the smallest captured size
/// that holds `tokens`, or `tokens` itself above the largest (eager).
pub(crate) fn graph_padded_tokens(sorted_sizes: &[u32], tokens: u32) -> u32 {
    match sorted_sizes.binary_search(&tokens) {
        Ok(_) => tokens,
        Err(index) => sorted_sizes.get(index).copied().unwrap_or(tokens),
    }
}

pub(crate) fn atomic<K, C, F>(
    prefix: &str,
    suffix: &str,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> std::result::Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> std::result::Result<K, BuildError>,
{
    let name = format!("{prefix}.{suffix}");
    Ok(Op::new(
        name.clone(),
        std::sync::Arc::new(build_kernel(name, config, bridge)?),
    ))
}

pub(crate) fn push<K>(op: &Op<K>, input: K::Input, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = op.kernel.eval(&input);
    ev.push(metrics, || input.clone().into());
}
