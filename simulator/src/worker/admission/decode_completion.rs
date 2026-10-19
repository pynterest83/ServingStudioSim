//! How many tokens one resident decode request retires per iteration.
//!
//! This is the one thing speculative decoding changes about a lifecycle: an
//! ordinary engine emits exactly one token per resident request per iteration,
//! a speculating engine verifies a fixed-width window and retires between one
//! and `draft_tokens + 1`. Everything else about admission — placement,
//! chunking, capacity gates, retraction — is identical, so the difference is a
//! sub-axis of the lifecycle rather than a second lifecycle.
//!
//! A policy owns exactly two facts:
//!
//! - [`DecodeCompletion::query_tokens_per_request`], the *fixed* query width the
//!   engine submits per resident request. It is a batch-shape fact the
//!   lifecycle needs before the iteration runs (to size its token budget) and
//!   the execution adapter needs to lower the L4 input. It does not depend on
//!   how many tokens are ultimately accepted.
//! - [`DecodeCompletion::complete_decodes`], the post-iteration transition:
//!   emit the retired tokens, stamp finished requests, and grow KV by exactly
//!   as many positions as were retired.
//!
//! It does not own membership, capacity, or batch composition. The lifecycle
//! decides *whether* a partition runs decode; this decides *how far* it moves.

use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, Time, UnifiedStage};
use crate::worker::kv::IterWorkerKv;
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};
use crate::worker::shared::context::WorkerContext;

/// Per-iteration decode advancement, as a lifecycle sub-axis.
///
/// Parameterized by the store rather than method-generic: the capability every
/// implementation needs — `IterWorkerKv` for decode membership plus its
/// `KvStore` supertrait for `advance` — is nameable at the implementation
/// boundary, so `ChunkedPrefillAdmission` carries `D` unbounded and states
/// `D: DecodeCompletion<K>` on its `IterAdmission<K>` impl.
pub trait DecodeCompletion<K: IterWorkerKv> {
    /// Query rows the engine submits per resident decode request each
    /// iteration. Fixed for the worker's lifetime; the token budget and the
    /// L4 input lowering both read it.
    fn query_tokens_per_request(&self) -> u32;

    /// Retire this iteration's accepted tokens on one partition, appending any
    /// request that finished to `completed`, and advance KV to match.
    ///
    /// `finished_by_accepted_drafts` receives the subset of `completed` that had
    /// more than one output token left before this round: an all-rejected
    /// verify, which still retires the target's own token, would not have
    /// finished them, so accepted drafts did. An engine that schedules one step
    /// ahead cannot rule out such a request's next round (see
    /// `ChunkedPrefillAdmission::with_async_scheduling`). One-token decode never
    /// reports any.
    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        finished_by_accepted_drafts: &mut Vec<RequestId>,
        now: Time,
    );
}

/// The ordinary engine: one query row in, one token out, every resident request
/// moving the same single position.
#[derive(Clone, Copy, Debug, Default)]
pub struct SingleTokenDecodeCompletion;

impl<K: IterWorkerKv> DecodeCompletion<K> for SingleTokenDecodeCompletion {
    fn query_tokens_per_request(&self) -> u32 {
        1
    }

    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        _finished_by_accepted_drafts: &mut Vec<RequestId>,
        now: Time,
    ) {
        {
            let mut store = context.requests.borrow_mut();
            kv_store.visit_decode_members(partition, |request, _| {
                let record = &mut store[request];
                record.record_token(now, context.log_tokens());
                if record.is_complete() {
                    context.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(request);
                }
            });
        }
        // Every member moved the same distance, so one whole-partition advance
        // is exact — no need to enumerate the membership a second time.
        kv_store.advance(AdvanceScope::WholePartition(partition), 1);
    }
}

/// Opt-in acceptance chains keyed by output position: the verify of a request
/// that has already emitted `n` output tokens draws from the chain of the first
/// band whose inclusive upper bound is `>= n` (the last band is open).
///
/// Measured MTP acceptance rises with the position inside the output (on the
/// agentic GLM passes, tokens per verify go from ~3.9 over the first 64 output
/// tokens to ~5.4 beyond 8k) and barely depends on the request's total length
/// once the position is fixed. One pooled chain per request matches the pooled
/// tokens per verify yet gives short requests the acceptance of long ones, so
/// their TPOT, and therefore the request-level TPOT percentiles, come out
/// optimistic. When set, the bands REPLACE the request's own `accept_rate`, so
/// pair them with pooled-chain traces, not per-request measured chains (which
/// already carry the position effect of that request's length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptancePositionBands {
    /// Inclusive upper bounds on output tokens emitted before the verify, one
    /// per band except the open last one; strictly increasing. Only the first
    /// `bands - 1` entries are live. Fixed capacity keeps the worker
    /// configuration `Copy`.
    max_output_tokens: [u32; MAX_ACCEPTANCE_BANDS - 1],
    /// One per-position conditional chain per band; only the first `bands`
    /// rows and `draft_tokens` columns are live.
    rates: [[f32; MAX_BAND_DRAFT_TOKENS]; MAX_ACCEPTANCE_BANDS],
    bands: usize,
    draft_tokens: usize,
}

/// Most output-position bands [`AcceptancePositionBands`] holds.
pub const MAX_ACCEPTANCE_BANDS: usize = 16;
/// Widest draft window [`AcceptancePositionBands`] holds.
pub const MAX_BAND_DRAFT_TOKENS: usize = 16;

impl AcceptancePositionBands {
    /// Build from the selector's flat lists: `rates` holds
    /// `(max_output_tokens.len() + 1) * draft_tokens` conditional probabilities,
    /// band-major. Both lists empty means the knob is off (`Ok(None)`).
    pub fn from_flat(
        max_output_tokens: &[u32],
        rates: &[f32],
        draft_tokens: u32,
    ) -> Result<Option<Self>, String> {
        if max_output_tokens.is_empty() && rates.is_empty() {
            return Ok(None);
        }
        let width = draft_tokens as usize;
        if width == 0 || width > MAX_BAND_DRAFT_TOKENS {
            return Err(format!(
                "acceptance position bands need a draft width in 1..={MAX_BAND_DRAFT_TOKENS}, \
                 got {width}"
            ));
        }
        let bands = max_output_tokens.len() + 1;
        if bands > MAX_ACCEPTANCE_BANDS {
            return Err(format!(
                "acceptance_position_band_max_tokens allows at most {} bounds, got {}",
                MAX_ACCEPTANCE_BANDS - 1,
                max_output_tokens.len()
            ));
        }
        if rates.len() != bands * width {
            return Err(format!(
                "acceptance_position_band_rates must hold {bands} bands x {width} draft \
                 positions = {} values (one band per acceptance_position_band_max_tokens \
                 bound plus the open last band), got {}",
                bands * width,
                rates.len()
            ));
        }
        if max_output_tokens.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(
                "acceptance_position_band_max_tokens must be strictly increasing".to_string(),
            );
        }
        if let Some(rate) = rates
            .iter()
            .find(|rate| !rate.is_finite() || !(0.0..=1.0).contains(*rate))
        {
            return Err(format!(
                "acceptance_position_band_rates must lie in [0, 1], got {rate}"
            ));
        }
        let mut out = Self {
            max_output_tokens: [0; MAX_ACCEPTANCE_BANDS - 1],
            rates: [[0.0; MAX_BAND_DRAFT_TOKENS]; MAX_ACCEPTANCE_BANDS],
            bands,
            draft_tokens: width,
        };
        out.max_output_tokens[..bands - 1].copy_from_slice(max_output_tokens);
        for (row, chain) in out.rates.iter_mut().zip(rates.chunks(width)) {
            row[..width].copy_from_slice(chain);
        }
        Ok(Some(out))
    }

    /// The conditional chain a verify draws from after `output_tokens_emitted`
    /// tokens, one probability per draft position.
    fn chain_for(&self, output_tokens_emitted: u32) -> &[f32] {
        let band = self.max_output_tokens[..self.bands - 1]
            .partition_point(|bound| *bound < output_tokens_emitted);
        &self.rates[band][..self.draft_tokens]
    }
}

/// Opt-in per-request acceptance spread: every request draws one odds
/// multiplier `m` (uniformly from the list, keyed by seed and request id, so it
/// is fixed for the request's life and independent of batching), and each draft
/// position's conditional probability `p` of its chain becomes
/// `p m / (1 - p + p m)`.
///
/// Measured per-request tokens per verify spread far wider than the per-step
/// Bernoulli noise of one shared chain (agentic GLM passes: sd ~0.8 vs ~0.2 at
/// 257-1024 output tokens), so the slowest-accepting requests set the TPOT
/// tail. The list is typically lognormal quantiles recentered so the pooled
/// tokens per verify is unchanged. Odds scaling needs no transcendental
/// function, keeping runs bit-identical across machines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptanceRequestSpread {
    multipliers: [f32; MAX_ACCEPTANCE_SPREAD],
    len: usize,
}

/// Most odds multipliers [`AcceptanceRequestSpread`] holds.
pub const MAX_ACCEPTANCE_SPREAD: usize = 64;

/// Decorrelates the per-request multiplier draw from the per-step draws.
const REQUEST_SPREAD_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

impl AcceptanceRequestSpread {
    /// Build from the selector's list; empty means the knob is off.
    pub fn from_list(multipliers: &[f32]) -> Result<Option<Self>, String> {
        if multipliers.is_empty() {
            return Ok(None);
        }
        if multipliers.len() > MAX_ACCEPTANCE_SPREAD {
            return Err(format!(
                "acceptance_request_odds_multipliers allows at most {MAX_ACCEPTANCE_SPREAD} \
                 values, got {}",
                multipliers.len()
            ));
        }
        if let Some(value) = multipliers
            .iter()
            .find(|value| !value.is_finite() || **value <= 0.0)
        {
            return Err(format!(
                "acceptance_request_odds_multipliers must be finite and > 0, got {value}"
            ));
        }
        let mut out = Self {
            multipliers: [1.0; MAX_ACCEPTANCE_SPREAD],
            len: multipliers.len(),
        };
        out.multipliers[..multipliers.len()].copy_from_slice(multipliers);
        Ok(Some(out))
    }

    /// The odds multiplier `request` keeps for its whole life under `seed`.
    fn multiplier(&self, seed: u64, request: RequestId) -> f64 {
        let key =
            (seed ^ REQUEST_SPREAD_SALT) ^ u64::from(request.0).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        f64::from(self.multipliers[(splitmix64(key) % self.len as u64) as usize])
    }
}

/// `p` with its odds scaled by `multiplier`; exact at 0 and 1.
fn scale_odds(p: f32, multiplier: f64) -> f32 {
    let p = f64::from(p);
    (p * multiplier / (1.0 - p + p * multiplier)) as f32
}

/// The speculating engine: a fixed `draft_tokens + 1` verify window per resident
/// request, retiring the target's own token plus the leading run of accepted
/// drafts.
pub struct SpeculativeDecodeCompletion {
    draft_tokens: u32,
    seed: u64,
    /// Opt-in output-position acceptance that replaces each request's own
    /// chain; `None` (the default) draws from the request's `accept_rate`.
    position_bands: Option<AcceptancePositionBands>,
    /// Opt-in per-request odds multiplier on every position probability;
    /// `None` (the default) leaves the chain as is.
    request_spread: Option<AcceptanceRequestSpread>,
    /// Scratch, indexed by retired count: `request_buckets[n]` holds the
    /// requests that retired `n` tokens this iteration. Requests in the same
    /// iteration advance by different distances, so KV takes one
    /// `RequestSubset` advance per distinct distance instead of one call each.
    request_buckets: Vec<Vec<RequestId>>,
}

impl SpeculativeDecodeCompletion {
    pub(crate) fn new(draft_tokens: u32, seed: u64) -> Self {
        assert!(
            draft_tokens > 0,
            "speculative draft_tokens must be positive"
        );
        // Indexed by retired count, which runs 1..=draft_tokens + 1.
        let bucket_count = draft_tokens
            .checked_add(2)
            .and_then(|count| usize::try_from(count).ok())
            .expect("speculative acceptance bucket count must fit usize");
        Self {
            draft_tokens,
            seed,
            position_bands: None,
            request_spread: None,
            request_buckets: vec![Vec::new(); bucket_count],
        }
    }

    /// Draw every verify from the output-position chains instead of the
    /// request's own `accept_rate` (see [`AcceptancePositionBands`]).
    pub(crate) fn with_position_bands(mut self, bands: Option<AcceptancePositionBands>) -> Self {
        self.position_bands = bands;
        self
    }

    /// Give every request a fixed odds multiplier on its chain (see
    /// [`AcceptanceRequestSpread`]).
    pub(crate) fn with_request_spread(mut self, spread: Option<AcceptanceRequestSpread>) -> Self {
        self.request_spread = spread;
        self
    }

    /// Tokens this verify retires: the target's own token, plus the leading run
    /// of accepted drafts, truncated by the request's remaining output length.
    fn accepted_length(
        &self,
        request: RequestId,
        output_tokens_emitted: u32,
        accept_rate: &AcceptanceProfile,
        remaining_output_tokens: u32,
    ) -> u32 {
        let band_chain = self
            .position_bands
            .as_ref()
            .map(|bands| bands.chain_for(output_tokens_emitted));
        let multiplier = self
            .request_spread
            .as_ref()
            .map(|spread| spread.multiplier(self.seed, request));
        let mut accepted = 1_u32;
        for draft_position in 0..self.draft_tokens {
            let position_rate = match band_chain {
                Some(chain) => chain[draft_position as usize],
                None => accept_rate.at_position(draft_position, self.draft_tokens),
            };
            let position_rate = match multiplier {
                Some(multiplier) => scale_odds(position_rate, multiplier),
                None => position_rate,
            };
            if !bernoulli(
                self.seed,
                request,
                output_tokens_emitted,
                draft_position,
                position_rate,
            ) {
                break;
            }
            accepted += 1;
        }
        accepted.min(remaining_output_tokens)
    }
}

impl<K: IterWorkerKv> DecodeCompletion<K> for SpeculativeDecodeCompletion {
    fn query_tokens_per_request(&self) -> u32 {
        self.draft_tokens + 1
    }

    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        finished_by_accepted_drafts: &mut Vec<RequestId>,
        now: Time,
    ) {
        for bucket in &mut self.request_buckets {
            bucket.clear();
        }
        {
            let mut store = context.requests.borrow_mut();
            kv_store.visit_decode_members(partition, |request, resident_kv| {
                let record = &mut store[request];
                let accept_rate = match &record.request.definition.decoding {
                    DecodingStrategy::Speculative { accept_rate } => accept_rate,
                    // Not a tolerated mode: the verify width is an engine-level
                    // constant, so a request with no declared acceptance would
                    // pay the wide batch and advance one position, which is not
                    // a thing this engine does. It is a composition error.
                    DecodingStrategy::Standard => {
                        panic!(
                            "speculative completion received standard request {}",
                            request.0
                        )
                    }
                };
                let remaining = record
                    .request
                    .definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted);
                assert!(
                    remaining > 0,
                    "completed request {} remained in speculative decode membership",
                    request.0
                );
                let accepted = self.accepted_length(
                    request,
                    record.progress.output_tokens_emitted,
                    accept_rate,
                    remaining,
                );
                let observation = record
                    .telemetry
                    .speculative
                    .get_or_insert_with(Default::default);
                observation.query_width = self.draft_tokens + 1;
                observation.decode_rounds += 1;
                observation.resident_kv_sum += resident_kv as u64;
                observation.emitted_tokens += u64::from(accepted);
                observation.pending_decode = None;
                for _ in 0..accepted {
                    record.record_token(now, context.log_tokens());
                }
                self.request_buckets[accepted as usize].push(request);
                if record.is_complete() {
                    context.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(request);
                    if remaining > 1 {
                        finished_by_accepted_drafts.push(request);
                    }
                }
            });
        }
        for accepted in 1..=self.draft_tokens + 1 {
            let request_ids = &self.request_buckets[accepted as usize];
            if !request_ids.is_empty() {
                kv_store.advance(
                    AdvanceScope::RequestSubset {
                        partition,
                        request_ids,
                    },
                    accepted,
                );
            }
        }
    }
}

/// One draft position's accept/reject draw.
///
/// Keyed, not sequential: the outcome is a pure function of
/// `(seed, request, output_tokens_emitted, draft_position)`, so a request's
/// acceptance chain does not depend on how it was batched, which partition it
/// landed on, or how many other requests drew before it. That is what makes a
/// scheduler A/B meaningful — acceptance is held fixed, so the difference is
/// attributable to the scheduler — and it is what lets an offline analysis
/// replay this sampler for one request without simulating the whole fleet.
fn bernoulli(
    seed: u64,
    request: RequestId,
    output_tokens_emitted: u32,
    draft_position: u32,
    accept_rate: f32,
) -> bool {
    if accept_rate <= 0.0 {
        return false;
    }
    if accept_rate >= 1.0 {
        return true;
    }
    let key = seed
        ^ u64::from(request.0).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(output_tokens_emitted).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ u64::from(draft_position).wrapping_mul(0x94D0_49BB_1331_11EB);
    let draw = splitmix64(key) >> 11;
    let unit = draw as f64 * (1.0 / ((1_u64 << 53) as f64));
    unit < f64::from(accept_rate)
}

/// splitmix64 finalizer. No transcendental ops and no carried state, so a run
/// is bit-identical across machines and across scheduling orders.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::{PoolId, SharedRequests, WorkerId};
    use crate::test_helpers::shared_with;
    use crate::worker::kv::{FullAttnKv, KvStore};

    const DRAFT_TOKENS: u32 = 3;

    #[test]
    fn multi_round_acceptance_matches_python_diagnostic_fixture() {
        #[derive(serde::Deserialize)]
        struct Chain {
            seed: u64,
            request_id: u32,
            output_len: u32,
            rates: Vec<f32>,
            emitted: Vec<u32>,
        }
        let chains: Vec<Chain> = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/speculative_acceptance_chains.json"
        ))
        .unwrap();
        for chain in chains {
            let policy = SpeculativeDecodeCompletion::new(chain.rates.len() as u32, chain.seed);
            let acceptance = AcceptanceProfile::ByPosition(chain.rates);
            let mut emitted = 1;
            let mut observed = Vec::new();
            while emitted < chain.output_len {
                let count = policy.accepted_length(
                    RequestId(chain.request_id),
                    emitted,
                    &acceptance,
                    chain.output_len - emitted,
                );
                observed.push(count);
                emitted += count;
            }
            assert_eq!(observed, chain.emitted);
        }
    }

    fn uniform(rate: f32) -> DecodingStrategy {
        DecodingStrategy::Speculative {
            accept_rate: AcceptanceProfile::Uniform(rate),
        }
    }

    fn by_position(rates: &[f32]) -> DecodingStrategy {
        DecodingStrategy::Speculative {
            accept_rate: AcceptanceProfile::ByPosition(rates.to_vec()),
        }
    }

    fn context_for(requests: SharedRequests) -> WorkerContext {
        WorkerContext {
            id: WorkerId(0),
            pool: PoolId(0),
            requests,
            log_output_token_times: false,
            log_stage_transitions: false,
        }
    }

    /// `count` resident decode requests that have each emitted their first
    /// token, all carrying `decoding` and `target_output_tokens` in total.
    ///
    /// `kv_decode_budget` is the KV-side remaining-decode count, which a test
    /// can raise above the request's own remaining length to isolate which of
    /// the two limits truncated a chain.
    fn resident_decodes(
        per_request: &[DecodingStrategy],
        target_output_tokens: u32,
        kv_decode_budget: u32,
    ) -> (FullAttnKv, WorkerContext) {
        let rows: Vec<(u32, u32, u32)> = (0..per_request.len() as u32)
            .map(|id| (id, 8, target_output_tokens))
            .collect();
        let requests = shared_with(&rows);
        {
            let mut store = requests.borrow_mut();
            for (id, decoding) in per_request.iter().enumerate() {
                let record = &mut store[RequestId(id as u32)];
                record.request.definition.decoding = decoding.clone();
                record.progress.prefill_tokens_processed = 8;
                record.record_token(Time::ZERO, false);
            }
        }
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        for id in 0..per_request.len() as u32 {
            let request = RequestId(id);
            let footprint = kv_store.footprint(request, 8, target_output_tokens);
            KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
            kv_store.commit_resident(request, 0, 8, kv_decode_budget);
        }
        (kv_store, context_for(requests))
    }

    fn one_resident_decode(
        target_output_tokens: u32,
        kv_decode_budget: u32,
        decoding: DecodingStrategy,
    ) -> (FullAttnKv, WorkerContext) {
        resident_decodes(&[decoding], target_output_tokens, kv_decode_budget)
    }

    /// Tokens emitted and KV positions grown by one iteration on request 0.
    /// They must agree: KV that outruns emission is exactly the silent
    /// over-allocation this policy exists to avoid.
    fn run_one_iteration(
        policy: &mut impl DecodeCompletion<FullAttnKv>,
        kv_store: &mut FullAttnKv,
        context: &WorkerContext,
    ) -> (u32, u64) {
        let before = context.requests.borrow()[RequestId(0)]
            .progress
            .output_tokens_emitted;
        let mut completed = Vec::new();
        policy.complete_decodes(
            kv_store,
            0,
            context,
            &mut completed,
            &mut Vec::new(),
            Time::from_ms(1.0),
        );
        let after = context.requests.borrow()[RequestId(0)]
            .progress
            .output_tokens_emitted;
        let mut grown = 0;
        kv_store.visit_decode_members(0, |request, current_kv| {
            if request == RequestId(0) {
                grown = current_kv - 8;
            }
        });
        (after - before, grown)
    }

    #[test]
    fn a_verify_window_is_wider_than_an_ordinary_decode_row() {
        assert_eq!(
            DecodeCompletion::<FullAttnKv>::query_tokens_per_request(&SingleTokenDecodeCompletion),
            1
        );
        assert_eq!(
            DecodeCompletion::<FullAttnKv>::query_tokens_per_request(
                &SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 0)
            ),
            DRAFT_TOKENS + 1,
            "the engine verifies every draft plus the target's own position"
        );
    }

    #[test]
    fn a_rejected_draft_still_retires_the_target_token() {
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(0.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (1, 1),
            "a verify pass always produces the target's token, so no request can stall"
        );
    }

    #[test]
    fn a_fully_accepted_window_retires_every_drafted_position() {
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(1.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (DRAFT_TOKENS + 1, u64::from(DRAFT_TOKENS + 1))
        );
    }

    #[test]
    fn a_per_position_profile_reads_each_positions_own_probability() {
        // The defect this catches: collapsing the vector to one number. A
        // measured acceptance curve is not geometric, and position 1 rejecting
        // must stop the chain even though position 2 would have accepted.
        let (mut kv_store, context) = one_resident_decode(64, 63, by_position(&[1.0, 0.0, 1.0]));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (2, 2),
            "target token plus position 0 only; the chain stops at the first reject"
        );
    }

    #[test]
    #[should_panic(expected = "must equal speculative draft_tokens")]
    fn a_profile_shallower_than_the_verify_width_is_refused_before_any_draw() {
        // The trace declares a depth and the worker declares a width. If they
        // disagree the run is not modelling what either of them says.
        let (mut kv_store, context) = one_resident_decode(64, 63, by_position(&[0.5, 0.5]));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        run_one_iteration(&mut policy, &mut kv_store, &context);
    }

    #[test]
    #[should_panic(expected = "received standard request")]
    fn a_request_with_no_declared_acceptance_is_a_composition_error() {
        // Not a tolerated mode. The verify width is engine-level, so such a
        // request would pay the wide batch and advance one position — which is
        // not something a speculating engine does to any request.
        let (mut kv_store, context) = one_resident_decode(64, 63, DecodingStrategy::Standard);
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        run_one_iteration(&mut policy, &mut kv_store, &context);
    }

    #[test]
    fn the_target_output_length_truncates_an_over_long_acceptance_chain() {
        // Two tokens in total, one already emitted at prefill: the window may
        // accept four, but only one token remains to emit. KV is deliberately
        // left with room for 63 more, so the request's own target is the only
        // thing that can truncate the chain.
        let (mut kv_store, context) = one_resident_decode(2, 63, uniform(1.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        let mut completed = Vec::new();
        let mut finished_by_accepted_drafts = Vec::new();
        policy.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            &mut finished_by_accepted_drafts,
            Time::from_ms(1.0),
        );
        assert_eq!(completed, vec![RequestId(0)]);
        assert!(
            finished_by_accepted_drafts.is_empty(),
            "one token left: even an all-rejected verify would have finished it"
        );
        let store = context.requests.borrow();
        assert_eq!(store[RequestId(0)].progress.output_tokens_emitted, 2);
        let mut grown = 0;
        kv_store.visit_decode_members(0, |_, current_kv| grown = current_kv - 8);
        assert_eq!(
            grown, 1,
            "KV must not grow past the tokens actually emitted"
        );
    }

    #[test]
    fn a_request_finished_by_accepted_drafts_is_reported_as_such() {
        // Four tokens in total, one emitted at prefill: three remain, so only
        // accepted drafts can finish it in one round. A rejecting twin with
        // the same length stays resident.
        let (mut kv_store, context) = resident_decodes(&[uniform(1.0), uniform(0.0)], 4, 63);
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        let mut completed = Vec::new();
        let mut finished_by_accepted_drafts = Vec::new();
        policy.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            &mut finished_by_accepted_drafts,
            Time::from_ms(1.0),
        );
        assert_eq!(completed, vec![RequestId(0)]);
        assert_eq!(finished_by_accepted_drafts, vec![RequestId(0)]);
    }

    #[test]
    fn requests_that_accept_different_lengths_each_grow_by_their_own_chain() {
        let requests: SharedRequests = shared_with(&[(0, 8, 64), (1, 8, 64)]);
        {
            let mut store = requests.borrow_mut();
            for (id, rate) in [(0u32, 0.0f32), (1, 1.0)] {
                let record = &mut store[RequestId(id)];
                record.request.definition.decoding = uniform(rate);
                record.progress.prefill_tokens_processed = 8;
                record.record_token(Time::ZERO, false);
            }
        }
        let context = context_for(requests);
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        for id in 0..2u32 {
            let request = RequestId(id);
            let footprint = kv_store.footprint(request, 8, 64);
            KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
            kv_store.commit_resident(request, 0, 8, 63);
        }

        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        let mut completed = Vec::new();
        policy.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            &mut Vec::new(),
            Time::from_ms(1.0),
        );

        let mut grown = Vec::new();
        kv_store.visit_decode_members(0, |request, current_kv| {
            grown.push((request, current_kv - 8))
        });
        assert_eq!(
            grown,
            vec![
                (RequestId(0), 1),
                (RequestId(1), u64::from(DRAFT_TOKENS + 1))
            ],
            "one bucketed advance per distinct chain length, insertion order preserved"
        );
    }

    #[test]
    fn a_requests_chain_does_not_depend_on_what_it_was_batched_with() {
        // The defect this catches: drawing from a sequential stream. The two
        // batches below differ only in how many draws the *earlier* requests
        // consume — all-reject spends one apiece, all-accept spends three — so
        // under a shared stream the last request would land on different draws
        // and its acceptance would shift with batch composition, partition
        // placement, and arrival order. Every scheduler comparison would then
        // be confounded by acceptance moving underneath it.
        let observed = |neighbour_rate: f32| {
            let (mut kv_store, context) = resident_decodes(
                &[
                    uniform(neighbour_rate),
                    uniform(neighbour_rate),
                    uniform(0.5),
                ],
                64,
                63,
            );
            let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 12345);
            let mut completed = Vec::new();
            policy.complete_decodes(
                &mut kv_store,
                0,
                &context,
                &mut completed,
                &mut Vec::new(),
                Time::from_ms(1.0),
            );
            let emitted = context.requests.borrow()[RequestId(2)]
                .progress
                .output_tokens_emitted;
            emitted
        };
        assert_eq!(
            observed(0.0),
            observed(1.0),
            "acceptance is keyed by (seed, request, progress, position), not by draw order"
        );
    }

    #[test]
    fn the_acceptance_draw_is_reproducible_and_seed_dependent() {
        let chain = |seed: u64| {
            let mut accepted = 0;
            for position in 0..DRAFT_TOKENS {
                if !bernoulli(seed, RequestId(3), 5, position, 0.5) {
                    break;
                }
                accepted += 1;
            }
            accepted
        };
        assert_eq!(chain(99), chain(99), "same seed reproduces the same chain");
        let spread: Vec<u32> = (0..64).map(chain).collect();
        assert!(
            spread.iter().any(|&length| length != spread[0]),
            "the seed must actually select a stream, not a constant"
        );
    }

    #[test]
    fn an_ordinary_engine_moves_every_resident_request_one_position() {
        let request = RequestId(0);
        let requests = shared_with(&[(0, 8, 64)]);
        {
            let mut store = requests.borrow_mut();
            store[request].progress.prefill_tokens_processed = 8;
        }
        let context = context_for(requests);
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        let footprint = kv_store.footprint(request, 8, 64);
        KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
        kv_store.commit_resident(request, 0, 8, 64);

        let mut completed = Vec::new();
        SingleTokenDecodeCompletion.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            &mut Vec::new(),
            Time::from_ms(1.0),
        );
        assert!(completed.is_empty());
        assert_eq!(
            context.requests.borrow()[request]
                .progress
                .output_tokens_emitted,
            1
        );
        let mut grown = 0;
        kv_store.visit_decode_members(0, |_, current_kv| grown = current_kv - 8);
        assert_eq!(grown, 1);
    }

    fn bands(max_output_tokens: &[u32], rates: &[f32]) -> AcceptancePositionBands {
        AcceptancePositionBands::from_flat(max_output_tokens, rates, DRAFT_TOKENS)
            .unwrap()
            .expect("non-empty lists build bands")
    }

    #[test]
    fn empty_position_band_lists_leave_the_knob_off() {
        assert_eq!(
            AcceptancePositionBands::from_flat(&[], &[], DRAFT_TOKENS),
            Ok(None)
        );
    }

    #[test]
    fn position_bands_select_by_tokens_already_emitted_with_inclusive_bounds() {
        let bands = bands(&[1, 4], &[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]);
        let first = |emitted| bands.chain_for(emitted)[0];
        assert_eq!(first(0), 0.1);
        assert_eq!(first(1), 0.1, "the bound is inclusive");
        assert_eq!(first(2), 0.2);
        assert_eq!(first(4), 0.2);
        assert_eq!(first(5), 0.3, "past the last bound the open band applies");
        assert_eq!(first(u32::MAX), 0.3);
    }

    #[test]
    fn malformed_position_bands_are_refused() {
        let build = |bounds: &[u32], rates: &[f32]| {
            AcceptancePositionBands::from_flat(bounds, rates, DRAFT_TOKENS)
        };
        assert!(build(&[8], &[0.5; 3]).unwrap_err().contains("2 bands x 3"));
        assert!(build(&[], &[0.5; 4]).is_err());
        assert!(build(&[8, 8], &[0.5; 9])
            .unwrap_err()
            .contains("strictly increasing"));
        assert!(build(&[8], &[0.5, 0.5, 0.5, 0.5, 1.5, 0.5])
            .unwrap_err()
            .contains("[0, 1]"));
        assert!(build(&[8], &[0.5, 0.5, 0.5, 0.5, f32::NAN, 0.5]).is_err());
        assert!(AcceptancePositionBands::from_flat(&[], &[0.5], 0).is_err());
        let too_many: Vec<u32> = (1..=MAX_ACCEPTANCE_BANDS as u32).collect();
        let rates = vec![0.5; (too_many.len() + 1) * DRAFT_TOKENS as usize];
        assert!(build(&too_many, &rates).unwrap_err().contains("at most"));
    }

    #[test]
    fn position_bands_replace_the_request_chain_by_output_position() {
        // The request's own chain accepts nothing; the first band (<= 1 token
        // emitted, i.e. the first verify after prefill) accepts every draft and
        // the open band accepts nothing again.
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(0.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7)
            .with_position_bands(Some(bands(&[1], &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])));
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (DRAFT_TOKENS + 1, u64::from(DRAFT_TOKENS + 1)),
            "one emitted token falls in the first band: the whole window is accepted"
        );
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (1, u64::from(DRAFT_TOKENS + 2)),
            "five emitted tokens fall in the open band: only the target token \
             (KV growth is cumulative over both verifies)"
        );
    }

    #[test]
    fn seven_wide_position_bands_retire_up_to_eight_tokens() {
        // MTP K = 7 (the 20261019_3 what-if): the band chains are 7 wide and a
        // verify retires the target token plus the leading accepted run.
        const K: u32 = 7;
        let seven = |rates: &[f32]| {
            AcceptancePositionBands::from_flat(&[], rates, K)
                .unwrap()
                .expect("non-empty lists build bands")
        };
        assert!(AcceptancePositionBands::from_flat(&[], &[0.9; 5], K)
            .unwrap_err()
            .contains("1 bands x 7"));
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(0.0));
        let mut policy = SpeculativeDecodeCompletion::new(K, 7)
            .with_position_bands(Some(seven(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])));
        assert_eq!(
            DecodeCompletion::<FullAttnKv>::query_tokens_per_request(&policy),
            K + 1
        );
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (6, 6),
            "draft 6 is rejected: the target token plus drafts 1-5"
        );
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(0.0));
        let mut policy =
            SpeculativeDecodeCompletion::new(K, 7).with_position_bands(Some(seven(&[1.0; 7])));
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (K + 1, u64::from(K + 1))
        );
    }

    #[test]
    fn absent_position_bands_draw_exactly_the_request_chain() {
        // Same seed and request: `with_position_bands(None)` must not touch the
        // keyed draws, so every existing run stays bit-identical.
        let draws = |policy: SpeculativeDecodeCompletion| {
            let (mut kv_store, context) = one_resident_decode(4096, 4095, uniform(0.7));
            let mut policy = policy;
            (0..200)
                .map(|_| run_one_iteration(&mut policy, &mut kv_store, &context).0)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            draws(SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 11)),
            draws(SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 11).with_position_bands(None))
        );
    }

    #[test]
    fn an_empty_spread_is_off_and_malformed_spreads_are_refused() {
        assert_eq!(AcceptanceRequestSpread::from_list(&[]), Ok(None));
        assert!(AcceptanceRequestSpread::from_list(&[1.0, 0.0])
            .unwrap_err()
            .contains("> 0"));
        assert!(AcceptanceRequestSpread::from_list(&[f32::INFINITY]).is_err());
        assert!(
            AcceptanceRequestSpread::from_list(&[1.0; MAX_ACCEPTANCE_SPREAD + 1])
                .unwrap_err()
                .contains("at most")
        );
    }

    #[test]
    fn odds_scaling_is_exact_at_the_ends_and_monotone() {
        assert_eq!(scale_odds(0.0, 3.0), 0.0);
        assert_eq!(scale_odds(1.0, 0.25), 1.0);
        assert!((scale_odds(0.5, 3.0) - 0.75).abs() < 1e-6, "odds 1 -> 3");
        assert!((scale_odds(0.8, 1.0) - 0.8).abs() < 1e-7);
        assert!(scale_odds(0.8, 0.5) < 0.8 && scale_odds(0.8, 2.0) > 0.8);
    }

    #[test]
    fn a_unit_spread_draws_exactly_the_request_chain() {
        // m = 1 is the identity on every probability, so the keyed draws, and
        // with them every retired count, are unchanged.
        let draws = |policy: SpeculativeDecodeCompletion| {
            let (mut kv_store, context) = one_resident_decode(4096, 4095, uniform(0.7));
            let mut policy = policy;
            (0..200)
                .map(|_| run_one_iteration(&mut policy, &mut kv_store, &context).0)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            draws(SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 11)),
            draws(
                SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 11)
                    .with_request_spread(AcceptanceRequestSpread::from_list(&[1.0, 1.0]).unwrap())
            )
        );
    }

    #[test]
    fn the_request_spread_is_fixed_per_request_and_scales_its_whole_chain() {
        // A huge multiplier turns p = 0.5 into ~1 for every position; a tiny
        // one into ~0. Each request keeps the same draw for its whole life, and
        // with two values both appear across enough requests.
        let spread = AcceptanceRequestSpread::from_list(&[1e9, 1e-9])
            .unwrap()
            .unwrap();
        let policy =
            SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 5).with_request_spread(Some(spread));
        let half = AcceptanceProfile::Uniform(0.5);
        let mut seen = [false; 2];
        for request in 0..64 {
            let first = policy.accepted_length(RequestId(request), 1, &half, 100);
            assert!(first == 1 || first == DRAFT_TOKENS + 1, "{first}");
            for emitted in [7, 50, 99] {
                assert_eq!(
                    policy.accepted_length(RequestId(request), emitted, &half, 100),
                    first,
                    "the multiplier must not change within a request"
                );
            }
            seen[usize::from(first == 1)] = true;
        }
        assert_eq!(seen, [true, true]);
    }
}
