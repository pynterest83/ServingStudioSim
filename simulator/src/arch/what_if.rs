//! Exploration-only counterfactual: scale the modeled time of selected kernel
//! slots (`what_if_kernel_time_scale` on the `speculative` worker selector).
//!
//! The question it answers is "how much end-to-end time would a faster kernel
//! buy?" before anyone profiles new L1 rows for that kernel: a fused-MoE
//! config that lifts the routed `down` GEMM from 60% to 75% of the HBM roof is
//! `{match: moe.routed.down, factor: 0.8}`. The scaled number is not a
//! measurement, so this knob breaks the "L1 is the only source of numbers"
//! rule on purpose and must never appear in a reproduction preset or an
//! alignment pack. Absent (the default), no wrapper is built and the worker
//! drives the arch model itself.
//!
//! Each rule's `match` is a dotted fragment of a compiled slot name
//! (`LeafDesc::name`, e.g. `unified.kda_moe.moe.routed.down`), compared on
//! whole `.`-separated segments: `moe.routed.down` hits `…moe.routed.down` but
//! not `…moe.routed.down_quant`. Rules are tried in order and the first that
//! matches a slot sets its factor.
//!
//! The factor multiplies the slot's `time_ms` after the arch fills its leaves
//! and before the CostTree folds them, so `Sum`/`Scale`/`Max`/`Parallel` see the
//! scaled child, the worker's multiplier and host-time floor see the scaled
//! iteration, and the `cost_log` row records the scaled slot times (folding
//! them still reproduces `total_time_ms`). FLOPs, bytes and energy are left as
//! measured.

use serde::Deserialize;

use crate::arch::contract::{GraphMode, SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::timing::{CostManifest, CostTree, FlatCostNode, LeafMetrics, SlotInput};

/// One `{match, factor}` rule of `what_if_kernel_time_scale`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelTimeScaleRule {
    /// Dotted slot-name fragment, matched on whole `.`-separated segments.
    #[serde(rename = "match")]
    pub pattern: String,
    /// Multiplier on every matched slot's modeled time; finite and > 0.
    pub factor: f64,
}

/// The rules resolved against one compiled CostTree: which slots scale by how
/// much, plus the flat nodes that fold the scaled slots.
#[derive(Clone, Debug)]
pub struct KernelTimeScale {
    /// `(slot, factor)` for every matched slot, in slot order.
    scaled: Vec<(usize, f32)>,
    nodes: Vec<FlatCostNode>,
}

impl KernelTimeScale {
    /// Resolve `rules` against `manifest`'s slot names. Rejects an empty rule
    /// list, an empty or malformed `match`, a factor that is not finite and
    /// positive, and a rule that sets no slot (it matches none, or only slots an
    /// earlier rule already claimed).
    pub fn resolve(rules: &[KernelTimeScaleRule], manifest: &CostManifest) -> Result<Self, String> {
        if rules.is_empty() {
            return Err("what_if_kernel_time_scale: no rules given".to_string());
        }
        for rule in rules {
            validate_rule(rule)?;
        }
        let mut claimed = vec![0usize; rules.len()];
        let mut matched = vec![false; rules.len()];
        let mut scaled = Vec::new();
        for (slot, desc) in manifest.slots.iter().enumerate() {
            let mut first = None;
            for (i, rule) in rules.iter().enumerate() {
                if segment_match(&desc.name, &rule.pattern) {
                    matched[i] = true;
                    first.get_or_insert(i);
                }
            }
            if let Some(i) = first {
                claimed[i] += 1;
                scaled.push((slot, rules[i].factor as f32));
            }
        }
        for (i, rule) in rules.iter().enumerate() {
            if !matched[i] {
                return Err(format!(
                    "what_if_kernel_time_scale: match {:?} hits no kernel slot of this model",
                    rule.pattern
                ));
            }
            if claimed[i] == 0 {
                return Err(format!(
                    "what_if_kernel_time_scale: match {:?} only hits slots an earlier rule \
                     already scales (the first matching rule wins)",
                    rule.pattern
                ));
            }
        }
        Ok(Self {
            scaled,
            nodes: manifest.nodes.clone(),
        })
    }

    /// The `(slot, factor)` pairs this scale applies, in slot order.
    pub fn scaled_slots(&self) -> &[(usize, f32)] {
        &self.scaled
    }

    /// Scale the matched slots of an evaluated leaf buffer in place and fold
    /// the buffer again through the CostTree.
    pub fn apply(&self, slots: &mut [LeafMetrics], scratch: &mut Vec<LeafMetrics>) -> LeafMetrics {
        for &(slot, factor) in &self.scaled {
            slots[slot].m.time_ms *= factor;
        }
        CostTree::aggregate(&self.nodes, slots, scratch)
    }
}

fn validate_rule(rule: &KernelTimeScaleRule) -> Result<(), String> {
    if rule.pattern.is_empty() {
        return Err("what_if_kernel_time_scale: a rule has an empty match".to_string());
    }
    if rule.pattern.split('.').any(str::is_empty) {
        return Err(format!(
            "what_if_kernel_time_scale: match {:?} has an empty `.` segment",
            rule.pattern
        ));
    }
    if !(rule.factor.is_finite() && rule.factor > 0.0) {
        return Err(format!(
            "what_if_kernel_time_scale: match {:?} has factor {}; it must be finite and > 0",
            rule.pattern, rule.factor
        ));
    }
    Ok(())
}

/// Whether `pattern` occurs in `name` starting and ending on a `.` boundary.
fn segment_match(name: &str, pattern: &str) -> bool {
    let (name, pattern) = (name.as_bytes(), pattern.as_bytes());
    if pattern.len() > name.len() {
        return false;
    }
    (0..=name.len() - pattern.len()).any(|at| {
        let end = at + pattern.len();
        &name[at..end] == pattern
            && (at == 0 || name[at - 1] == b'.')
            && (end == name.len() || name[end] == b'.')
    })
}

/// A speculative model whose evaluated slot times pass through a
/// [`KernelTimeScale`] before the fold. Every other query forwards unchanged.
pub struct WhatIfKernelTimeScaled<M> {
    inner: M,
    scale: KernelTimeScale,
}

impl<M: SpeculativeUnifiedModel> WhatIfKernelTimeScaled<M> {
    pub fn new(inner: M, rules: &[KernelTimeScaleRule]) -> Result<Self, String> {
        let scale = KernelTimeScale::resolve(rules, &inner.cost_log_manifest())?;
        Ok(Self { inner, scale })
    }
}

impl<M: SpeculativeUnifiedModel> SpeculativeUnifiedModel for WhatIfKernelTimeScaled<M> {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.inner.check_input(batch)
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.inner.eval_speculative_iter(batch, slots, scratch);
        self.scale.apply(slots, scratch)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.inner
            .eval_speculative_iter_with_inputs(batch, slots, scratch, inputs);
        self.scale.apply(slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.inner.cost_log_manifest()
    }

    fn graph_mode(&self, batch: &SpeculativeArchInput) -> Option<GraphMode> {
        self.inner.graph_mode(batch)
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.inner.total_kv_bytes_per_token()
    }

    fn max_model_len(&self) -> u32 {
        self.inner.max_model_len()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.inner.gpus_per_replica()
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.inner.num_attn_dp_groups()
    }

    fn num_attn_shards(&self) -> u16 {
        self.inner.num_attn_shards()
    }

    fn drafting_slots_per_request(&self) -> u32 {
        self.inner.drafting_slots_per_request()
    }

    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.inner.recurrent_state_bytes_per_request()
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.inner.recurrent_checkpoint_interval_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};
    use crate::common::{Time, WorkerId};
    use crate::timing::kernels::RmsNormKernelInput;
    use crate::timing::{CostNode, CostTreeBuilder, Metrics4};
    use crate::worker::config::{HostFloorMs, HostTimeModel};
    use crate::worker::CostBuffers;

    fn rule(pattern: &str, factor: f64) -> KernelTimeScaleRule {
        KernelTimeScaleRule {
            pattern: pattern.to_string(),
            factor,
        }
    }

    fn leaf(time_ms: f32) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms,
                flops: 1.0,
                bytes: 2.0,
                energy_j: 3.0,
            },
            ..LeafMetrics::ZERO
        }
    }

    /// One MoE layer in miniature: input quant, then the routed experts on an
    /// aux stream beside the shared expert, then the routed down projection's
    /// quant and GEMM on a two-rank `Max`, all folded `Scale{2}`.
    ///
    /// ```text
    /// Scale{2}( Sum( quant,
    ///                Parallel{1.0}( routed.gate_up, shared_expert.gate_up ),
    ///                Max{0.5}( routed.down_quant, routed.down ) ) )
    /// ```
    fn moe_tree() -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut leaf = |name: &str| b.leaf(name, "k", serde_json::Value::Null);
        let quant = leaf("m.layer.moe.routed.input_quant");
        let routed_gate_up = leaf("m.layer.moe.routed.gate_up");
        let shared_gate_up = leaf("m.layer.moe.shared_expert.gate_up");
        let down_quant = leaf("m.layer.moe.routed.down_quant");
        let down = leaf("m.layer.moe.routed.down");
        let root = CostNode::Scale {
            n: 2,
            child: Box::new(CostNode::Sum(vec![
                quant,
                CostNode::Parallel {
                    overlap: 1.0,
                    children: vec![routed_gate_up, shared_gate_up],
                },
                CostNode::Max {
                    overlap: 0.5,
                    children: vec![down_quant, down],
                },
            ])),
        };
        b.finish(root)
    }

    /// quant 1, routed gate_up 10, shared gate_up 9, down_quant 2, down 4.
    fn moe_slots() -> Vec<LeafMetrics> {
        [1.0, 10.0, 9.0, 2.0, 4.0].map(leaf).to_vec()
    }

    #[test]
    fn a_matched_slot_scales_and_an_unmatched_one_does_not() {
        let manifest = moe_tree().manifest();
        let scale = KernelTimeScale::resolve(&[rule("moe.routed.down", 0.75)], &manifest).unwrap();
        // Segment matching: `routed.down` is not a prefix hit on `down_quant`.
        assert_eq!(scale.scaled_slots(), &[(4, 0.75)]);

        let mut slots = moe_slots();
        scale.apply(&mut slots, &mut Vec::new());
        let times: Vec<f32> = slots.iter().map(|s| s.m.time_ms).collect();
        assert_eq!(times, vec![1.0, 10.0, 9.0, 2.0, 3.0]);
        // Only time moves: work and energy stay as measured.
        assert_eq!(slots[4].m.flops, 1.0);
        assert_eq!(slots[4].m.bytes, 2.0);
        assert_eq!(slots[4].m.energy_j, 3.0);
    }

    #[test]
    fn overlap_nodes_fold_the_scaled_child() {
        let tree = moe_tree();
        let manifest = tree.manifest();
        let flat = tree.flatten();
        let mut scratch = Vec::new();
        let baseline = CostTree::aggregate(&flat, &moe_slots(), &mut scratch)
            .m
            .time_ms;
        // 2 x (1 + max(10, 9) + max(2, 4) / 0.5)
        assert_eq!(baseline, 2.0 * (1.0 + 10.0 + 8.0));

        // Halving the routed gate_up drops it below the shared expert, so the
        // `Parallel` now takes the shared branch: 2 x (1 + 9 + 3 / 0.5).
        let scale = KernelTimeScale::resolve(
            &[
                rule("moe.routed.gate_up", 0.5),
                rule("moe.routed.down", 0.75),
            ],
            &manifest,
        )
        .unwrap();
        let mut slots = moe_slots();
        let total = scale.apply(&mut slots, &mut scratch);
        assert_eq!(total.m.time_ms, 2.0 * (1.0 + 9.0 + 6.0));
        // Folding the scaled slots again reproduces the returned total, as a
        // cost_log consumer would.
        let refolded = CostTree::aggregate(&flat, &slots, &mut scratch);
        assert_eq!(refolded.m.time_ms, total.m.time_ms);
        // Work still sums over every child.
        assert_eq!(total.m.flops, 2.0 * 5.0);
    }

    #[test]
    fn the_first_matching_rule_wins() {
        let manifest = moe_tree().manifest();
        let scale = KernelTimeScale::resolve(
            &[rule("routed.down", 0.5), rule("moe.routed", 0.25)],
            &manifest,
        )
        .unwrap();
        assert_eq!(
            scale.scaled_slots(),
            &[(0, 0.25), (1, 0.25), (3, 0.25), (4, 0.5)]
        );
    }

    #[test]
    fn invalid_rules_are_rejected_with_the_offending_match() {
        let manifest = moe_tree().manifest();
        let error =
            |rules: &[KernelTimeScaleRule]| KernelTimeScale::resolve(rules, &manifest).unwrap_err();
        assert!(error(&[]).contains("no rules"));
        assert!(error(&[rule("", 0.8)]).contains("empty match"));
        assert!(error(&[rule("moe..down", 0.8)]).contains("empty `.` segment"));
        assert!(error(&[rule(".down", 0.8)]).contains("empty `.` segment"));
        for factor in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            let message = error(&[rule("moe.routed.down", factor)]);
            assert!(message.contains("\"moe.routed.down\""), "{message}");
            assert!(message.contains("finite and > 0"), "{message}");
        }
        let missing = error(&[rule("moe.routed.down", 0.8), rule("moe.routed.gate", 0.9)]);
        assert!(missing.contains("\"moe.routed.gate\""), "{missing}");
        assert!(missing.contains("hits no kernel slot"), "{missing}");
        let shadowed = error(&[rule("moe.routed", 0.8), rule("routed.down", 0.9)]);
        assert!(shadowed.contains("\"routed.down\""), "{shadowed}");
        assert!(shadowed.contains("earlier rule"), "{shadowed}");
    }

    #[test]
    fn the_rule_parses_from_its_yaml_spelling_only() {
        let rules: Vec<KernelTimeScaleRule> =
            serde_yaml::from_str("- {match: moe.routed.down, factor: 0.8}\n").unwrap();
        assert_eq!(rules, vec![rule("moe.routed.down", 0.8)]);
        assert!(serde_yaml::from_str::<Vec<KernelTimeScaleRule>>(
            "- {match: moe.routed.down, factor: 0.8, scale: 2}\n"
        )
        .is_err());
    }

    /// A speculative model over [`moe_tree`] that answers every query with a
    /// value distinct from the trait default, so a wrapper that forgot to
    /// forward one would be caught.
    struct FakeModel;

    impl SpeculativeUnifiedModel for FakeModel {
        fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
            if batch.draft_tokens == 3 {
                Ok(())
            } else {
                Err("draft_tokens".to_string())
            }
        }

        fn eval_speculative_iter(
            &self,
            _batch: &SpeculativeArchInput,
            slots: &mut Vec<LeafMetrics>,
            scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            *slots = moe_slots();
            CostTree::aggregate(&moe_tree().flatten(), slots, scratch)
        }

        fn eval_speculative_iter_with_inputs(
            &self,
            batch: &SpeculativeArchInput,
            slots: &mut Vec<LeafMetrics>,
            scratch: &mut Vec<LeafMetrics>,
            inputs: &mut Vec<SlotInput>,
        ) -> LeafMetrics {
            inputs.clear();
            inputs.extend((0..5).map(|m| RmsNormKernelInput { m }.into()));
            self.eval_speculative_iter(batch, slots, scratch)
        }

        fn cost_log_manifest(&self) -> CostManifest {
            moe_tree().manifest()
        }

        fn graph_mode(&self, _batch: &SpeculativeArchInput) -> Option<GraphMode> {
            Some(GraphMode::FullGraph)
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            11
        }

        fn max_model_len(&self) -> u32 {
            12
        }

        fn gpus_per_replica(&self) -> u16 {
            8
        }

        fn num_attn_dp_groups(&self) -> u16 {
            2
        }

        fn num_attn_shards(&self) -> u16 {
            3
        }

        fn drafting_slots_per_request(&self) -> u32 {
            4
        }

        fn recurrent_state_bytes_per_request(&self) -> u64 {
            13
        }

        fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
            14
        }
    }

    fn decode_batch() -> SpeculativeArchInput {
        SpeculativeArchInput {
            draft_tokens: 3,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: 4,
                decode_tokens: 4,
                decode_requests: vec![SpeculativeDecodeInput {
                    kv_len: 10,
                    query_len: 4,
                }],
                total_kv_len: 6,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn scaled_model() -> WhatIfKernelTimeScaled<FakeModel> {
        WhatIfKernelTimeScaled::new(
            FakeModel,
            &[
                rule("moe.routed.gate_up", 0.5),
                rule("moe.routed.down", 0.75),
            ],
        )
        .unwrap()
    }

    #[test]
    fn the_wrapper_scales_both_eval_paths_and_forwards_every_other_query() {
        let model = scaled_model();
        let batch = decode_batch();
        let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());

        let total = model.eval_speculative_iter(&batch, &mut slots, &mut scratch);
        assert_eq!(total.m.time_ms, 2.0 * (1.0 + 9.0 + 6.0));
        assert_eq!(slots[1].m.time_ms, 5.0);
        assert_eq!(slots[4].m.time_ms, 3.0);

        let logged =
            model.eval_speculative_iter_with_inputs(&batch, &mut slots, &mut scratch, &mut inputs);
        assert_eq!(logged.m.time_ms, total.m.time_ms);
        assert_eq!(inputs.len(), 5);

        assert_eq!(model.cost_log_manifest(), FakeModel.cost_log_manifest());
        assert!(model.check_input(&batch).is_ok());
        assert!(model.check_input(&SpeculativeArchInput::default()).is_err());
        assert_eq!(model.graph_mode(&batch), Some(GraphMode::FullGraph));
        assert_eq!(model.total_kv_bytes_per_token(), 11);
        assert_eq!(model.max_model_len(), 12);
        assert_eq!(model.gpus_per_replica(), 8);
        assert_eq!(model.num_attn_dp_groups(), 2);
        assert_eq!(model.num_attn_shards(), 3);
        assert_eq!(model.drafting_slots_per_request(), 4);
        assert_eq!(model.recurrent_state_bytes_per_request(), 13);
        assert_eq!(model.recurrent_checkpoint_interval_tokens(), 14);
    }

    /// The worker's step clock (`kernel x multiplier + host fixed`) runs on the
    /// scaled iteration, not on a post-scaled total.
    #[test]
    fn the_scaled_kernel_time_reaches_the_worker_clock() {
        let host = HostTimeModel {
            fixed_ms: 1.0,
            floor_ms: HostFloorMs {
                full_graph: 0.0,
                piecewise: 0.0,
                eager_prefill: 0.0,
                eager_decode: 0.0,
            },
        };
        let step_ms = |model: &dyn SpeculativeUnifiedModel| {
            let mut cost = CostBuffers::new(
                None,
                "test",
                WorkerId(0),
                &crate::timing::CostManifestDoc::single("iter", model.cost_log_manifest()),
                1.5,
            )
            .with_host_time(Some(host));
            cost.run_speculative_iter(model, &decode_batch(), 0, Time::ZERO)
        };
        assert_eq!(
            step_ms(&FakeModel),
            Time::from_ms(2.0 * (1.0 + 10.0 + 8.0) * 1.5 + 1.0)
        );
        assert_eq!(
            step_ms(&scaled_model()),
            Time::from_ms(2.0 * (1.0 + 9.0 + 6.0) * 1.5 + 1.0)
        );
    }
}
