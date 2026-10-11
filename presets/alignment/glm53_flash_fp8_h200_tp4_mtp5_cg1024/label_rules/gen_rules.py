"""Generate the GLM-5.3-Flash FP8 H200 TP4 MTP-5 label rules (one order-free set).

Operations are named after the simulated slot suffix they own. A forward
operation claims its suffix in every layer tag (first_kda_dense, kda_dense,
dsa_moe, kda_moe) but never the MTP draft copy of the same suffix
(`excluded_slot_prefixes: unified.mtp.`); a draft operation (`mtp.*`,
phase `draft`) claims the step-0 and the recurrent-step copy together, so a
measured draft kernel is compared with its slot summed over all K steps.

Usage: python gen_rules.py target_rules.json

The checked-in target_rules.json is this output pruned to the rules that match
at least one position of the case-03 fixpoint (see manifest.json).
"""
import json
import sys

SRC = "logs/20261007_0_glm53_flash_h200_campaign/03_balanced_anchor_c32"
# FLASHINFER_MLA_SPARSE_SM90 capture of case 10 (vLLM picks it first when nvcc is on PATH;
# the arch's sparse_mla_backend: flashinfer_mla_sparse_sm90). Its kernels never co-occur
# with FA3's in one capture.
FI_SRC = "logs/20261010_1_glm53_h200_tock1/runs/Tn/10_full_saturation_rate"
MTP = "unified.mtp."
rules = []
ROLE = {}


def op(operation, role, slots, typ="model"):
    ROLE[operation] = (role, list(slots), typ)


def M(operation, name, **ev):
    """One rule for `operation`. Draft operations (`mtp.*`) default to phase
    draft and claim mtp.step_0.<s> + mtp.recurrent.<s>; forward operations
    default to phase forward and exclude the draft copies."""
    role, slots, typ = ROLE[operation]
    draft = operation.startswith("mtp.")
    r = {"operation": operation, "name": name, "type": typ, "role": role,
         "source_experiment": SRC}
    if draft:
        r["slot_suffixes"] = [f"mtp.{step}.{s}" for s in slots for step in ("step_0", "recurrent")]
    else:
        r["slot_suffixes"] = list(slots)
        r["excluded_slot_prefixes"] = [MTP]
    if typ == "collective":
        r["cross_rank"] = "synchronizing"
    ev.setdefault("phase", "draft" if draft else "forward")
    r.update({k: v for k, v in ev.items() if v is not None})
    rules.append(r)


def U(operation, name, note, **ev):
    """An explicit unmapped synchronizing collective (no simulated owner)."""
    r = {"operation": operation, "name": name, "status": "unmapped",
         "cross_rank": "synchronizing", "note": note, "source_experiment": SRC}
    r.update({k: v for k, v in ev.items() if v is not None})
    rules.append(r)


NVJ = "nvjet_sm90_tst_"          # cuBLASLt BF16 tiles (projections, lm_head)
TSS = "nvjet_sm90_tss_"          # only the MoE router GEMM in this capture
NVJ_ANY = "nvjet_sm90_"
NNT = "_NNT"                     # the W_UK absorb bmm (transposed B)
SKR = "cublasLt::splitKreduce_kernel"
EW = "at::native::elementwise_kernel"
VEW = "at::native::vectorized_elementwise_kernel"
UEW = "at::native::unrolled_elementwise_kernel"
EWI = "elementwise_kernel_with_index"
IDX = "at::native::index_elementwise_kernel"
COPY_U = "unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
COPY_BF16 = "elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
GATHER = "vectorized_gather_kernel<(int)16, long>"
BIGFUSE = "mhc_pre_big_fuse_with_norm_tilelang_kernel"
# (bool)1 = column-major (DeepGEMM) scales: dense FFN and shared expert;
# (bool)0 = row-major scales for the Triton fused MoE.
QUANT_DG = "per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, (bool)1"
QUANT_MOE = "per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, (bool)0"
DG = "deep_gemm::sm90_fp8_gemm_1d2d_impl"
ACT = "vllm::act_and_mul_kernel"
FADD_NORM = "vllm::fused_add_rms_norm_kernel"
ALL_REDUCES = ("twoshotAllreduceKernel", "oneshotAllreduceFusionKernel", "multimem_all_reduce_kernel",
               "cross_device_reduce_2stage", "ncclDevKernel_AllReduce_Sum_bf16_RING_LL")
GLUE_NAMES = (EW, VEW, UEW, EWI)

# ======================================================================
# forward: prologue / epilogue / boundaries
# ======================================================================
op("embedding", "vocab-parallel embedding lookup", ["embedding"])
op("hc_expand", "hyper-connection expand copy to [T,4,H]", ["hc_expand"])
op("tp.all_reduce",
   "TP all-reduce after embedding, attention and FFN: FlashInfer trtllm one-/two-shot up to its "
   "fusion workspace, then vLLM custom all-reduce, torch symm_mem multimem or NCCL ring by size",
   ["embedding_all_reduce.fusion", "embedding_all_reduce.fallback",
    "attn_all_reduce.fusion", "attn_all_reduce.fallback",
    "ffn_all_reduce.fusion", "ffn_all_reduce.fallback"], "collective")
op("mhc.boundary", "mHC boundary: mhc_post, tf32 hc_prenorm GEMM and fused pre/RMSNorm between sublayers",
   ["attn_mhc_pre", "attn_mhc_post_pre", "ffn_mhc_post_pre", "final_mhc_post"])
op("hc_contract_mean", "hyper-connection contract x.mean(dim=1)", ["hc_contract_mean"])
op("final_norm", "final RMSNorm", ["final_norm"])
op("lm_head", "tensor-parallel language-model head projection", ["lm_head"])

M("embedding", "vocab_parallel_embedding_kernel")
for n in ALL_REDUCES:
    M("tp.all_reduce", n)
M("hc_expand", COPY_BF16, after="tp.all_reduce", before_name="hc_prenorm_gemm")
for n in ("mhc_post_tilelang_kernel", "hc_prenorm_gemm", BIGFUSE, "mhc_fused_tilelang_kernel"):
    M("mhc.boundary", n)
M("hc_contract_mean", "ReduceOp<c10::BFloat16, at::native::MeanOps", after="mhc.boundary")
M("final_norm", "vllm::rms_norm_kernel")
M("lm_head", NVJ_ANY, phase="sample")
M("lm_head", SKR, phase="sample", after="lm_head")
U("sample.logits.tp_allgather", "ncclDevKernel_AllGather_RING_LL",
  "vocab-parallel logits all-gather before sampling; the unified CostTree has no sampling leaf",
  phase="sample")

# ======================================================================
# forward: KDA (34 layers)
# ======================================================================
K = "kda."
for s, role in [
        ("in_proj", "KDA merged q|k|v|beta|f_a|g_a BF16 projection"),
        ("f_b_proj", "KDA f_b low-rank gate up projection"),
        ("g_b_proj", "KDA g_b low-rank output-gate projection"),
        ("split_select", "KDA mixed-step split of the projections into verify / prefill rows (index_select x6)"),
        ("short_conv_verify", "KDA causal conv update over the K+1 verify rows"),
        ("short_conv_prefill", "KDA varlen causal conv over the prefill rows"),
        ("recurrent_verify", "KDA fused recurrent update over the K+1 verify rows"),
        ("state_gather", "KDA prefill SSM state gather"),
        ("chunk_prefill", "FlashKDA prefill call: q/k/v contiguous copies, beta transpose, prepare, recurrence"),
        ("state_scatter", "KDA prefill SSM state scatter"),
        ("merge_copy", "KDA mixed-step scatter of the two core outputs back into one buffer (index_copy x2)"),
        ("gated_norm", "KDA sigmoid-gated RMSNorm"),
        ("o_proj", "KDA output projection")]:
    op(K + s, role, [K + s])

M(K + "in_proj", NVJ, after_name=BIGFUSE, before_name=NVJ)
M(K + "f_b_proj", NVJ, after=K + "in_proj")
M(K + "g_b_proj", NVJ, after=K + "f_b_proj")
# folded bodies that open on the KDA projections: anchor on what follows them
M(K + "g_b_proj", NVJ, before_name="_causal_conv1d")
M(K + "g_b_proj", NVJ, before=K + "split_select")
M(K + "f_b_proj", NVJ, before=K + "g_b_proj")
M(K + "in_proj", NVJ, before=K + "f_b_proj")
M(K + "split_select", GATHER, after=K + "g_b_proj")
M(K + "split_select", GATHER, after=K + "split_select")
M(K + "split_select", GATHER, before_name="_causal_conv1d")
M(K + "split_select", GATHER, before=K + "split_select")
# index_select takes the small-index kernel for some of the six at small sizes
for prev in (K + "g_b_proj", K + "split_select"):
    M(K + "split_select", "indexSelectSmallIndex<c10::BFloat16", after=prev)
M(K + "split_select", "indexSelectSmallIndex<c10::BFloat16", before=K + "split_select")
# the six folded into a one-kernel repeat body (no neighbour inside the body);
# no other lone gather body occurs in the forward phase
M(K + "split_select", GATHER, after="<none>", before="<none>")
M(K + "split_select", "indexSelectSmallIndex<c10::BFloat16", after="<none>", before="<none>")
M(K + "short_conv_verify", "_causal_conv1d_update_kernel")
M(K + "short_conv_prefill", "_causal_conv1d_fwd_kernel")
M(K + "recurrent_verify", "fused_recurrent_gated_delta_rule_fwd_kernel")
M(K + "state_gather", "_gather_initial_states_kernel")
# FlashKDA's three .contiguous() copies and its beta transpose belong to the call
M(K + "chunk_prefill", COPY_BF16, after=K + "state_gather")
M(K + "chunk_prefill", COPY_BF16, after=K + "chunk_prefill")
M(K + "chunk_prefill", COPY_BF16, before_name="_flash_kda_fwd_prepare")
M(K + "chunk_prefill", COPY_BF16, before=K + "chunk_prefill")
# the four copies folded into a one-kernel repeat body
M(K + "chunk_prefill", COPY_BF16, after="<none>", before="<none>")
M(K + "chunk_prefill", "_flash_kda_fwd_prepare")
M(K + "chunk_prefill", "_flash_kda_fwd_recurrence")
M(K + "state_scatter", "_scatter_states_kernel")
M(K + "merge_copy", "index_copy_kernel_impl")
M(K + "gated_norm", "layer_norm_gated_fwd_kernel")
M(K + "o_proj", NVJ, after=K + "gated_norm")

# ======================================================================
# forward: DSA (11 layers)
# ======================================================================
D = "dsa."
for s, role in [
        ("fused_qkv_a", "DSA fused Q and latent KV projection"),
        ("q_kv_a_norm", "DSA fused q/kv RMSNorm"),
        ("q_b_proj", "DSA query up projection"),
        ("indexer.wq_b", "DSA indexer query projection"),
        ("indexer.wk_weights", "DSA indexer key/weights projection"),
        ("indexer.head_weights", "DSA indexer fp32 head-weights torch.mm (cast copy, SIMT/xmma sgemm, split-K reduce)"),
        ("indexer.k_norm", "DSA indexer k LayerNorm (Inductor-fused)"),
        ("indexer.q_fwht_quant", "DSA indexer FWHT-128 + FP8 quant of q"),
        ("indexer.weight_scale", "DSA indexer head weights x q scale (Inductor-fused)"),
        ("indexer.kpool_gate_score", "DSA indexer kpool gate-score projection"),
        ("indexer.prefill_gather", "DSA prefill k[idx] / gate_score[idx] gathers"),
        ("indexer.kpool_decode_update", "DSA kpool decode cache update"),
        ("indexer.kpool_prefill_write", "DSA kpool prefill softmax/rotate cache write"),
        ("indexer.kpool_tail_seed", "DSA kpool prefill tail seed"),
        ("indexer.mqa_logits_decode", "DSA indexer paged MQA logits, decode rows flattened to next_n=1"),
        ("indexer.topk_decode", "DSA indexer decode top-k (filtered above 32 rows, cooperative at <=64)"),
        ("indexer.mqa_logits_prefill", "DSA indexer prefill MQA logits with its k-cache gather"),
        ("indexer.topk_prefill", "DSA indexer prefill top-k"),
        ("indexer.expand_pools", "DSA pool-to-token expansion"),
        ("sparse_mla.mla_cache_append", "MLA BF16 latent cache append"),
        ("dense_prefill.kv_b_proj", "dense-MHA prefill kv_b up projection"),
        ("q_absorb", "MLA W_UK query absorb bmm"),
        ("sparse_mla.index_remap", "sparse MLA request-to-global index remap"),
        ("v_up", "MLA W_UV value up bmm"),
        ("o_proj", "DSA output projection")]:
    op(D + s, role, [D + s])
op(D + "dense_prefill.attention", "dense-MHA prefill FlashAttention-3 varlen call (prepare + mainloop)",
   [D + "dense_prefill.attention"])
# The sparse-MLA backend's per-call launch ahead of the attention has a backend-specific
# slot, and the cost manifest carries only the selected backend's. Each operation lists
# both so the one rule set resolves against either manifest (a rule body is resolved
# whether or not it fires); the kernel names pick the operation.
PREP_SLOTS = [D + "sparse_mla.q_rope_zero_fill", D + "sparse_mla.pack_indices"]
op(D + "sparse_mla.q_rope_zero_fill", "zero fill of FA3's 64-wide rope-free query", PREP_SLOTS)
op(D + "sparse_mla.pack_indices", "FLASHINFER_MLA_SPARSE_SM90 pack of each row's valid top-k "
   "slots into the planned CSR page table (_pack_topk_indices)", PREP_SLOTS)
op(D + "sparse_mla", "one sparse-MLA call per layer: FLASH_ATTN_MLA_SPARSE's FA3 call (prepare_varlen, "
   "mainloop, combine) or FLASHINFER_MLA_SPARSE_SM90's BatchMLAPageAttentionHopperKernel",
   [D + "sparse_mla.attention"])
op(D + "glue", "DSA index/mask plumbing around the indexer (fills, copies, aranges, compares), "
   "with the prefill-only plumbing", [D + "glue", D + "indexer_glue", D + "prefill_glue"])


def dsa_chain(P):
    """DSA rules shared by the forward layers and the MTP draft layer."""
    q = P + "dsa."
    M(q + "fused_qkv_a", NVJ, before=q + "q_kv_a_norm")
    M(q + "fused_qkv_a", SKR, before=q + "q_kv_a_norm")
    M(q + "fused_qkv_a", NVJ, before=q + "fused_qkv_a")
    if P:
        # the draft's anonymous Triton launches also include step glue
        M(q + "q_kv_a_norm", "kernel", name_exact=True, after_name=NVJ)
        M(q + "q_kv_a_norm", "kernel", name_exact=True, after_name=SKR)
    else:
        # the only anonymous Triton launch of the forward phase
        M(q + "q_kv_a_norm", "kernel", name_exact=True)
    M(q + "q_b_proj", NVJ, after=q + "q_kv_a_norm")
    M(q + "indexer.wq_b", NVJ, after=q + "q_b_proj")
    M(q + "indexer.wk_weights", NVJ, after=q + "indexer.wq_b")
    M(q + "indexer.head_weights", COPY_U, after=q + "indexer.wk_weights")
    M(q + "indexer.head_weights", "cutlass_80_simt_sgemm")
    M(q + "indexer.head_weights", "sm80_xmma_gemm_f32f32_f32f32_f32")
    M(q + "indexer.k_norm", "triton_per_fused__to_copy_native_layer_norm_0")
    M(q + "indexer.q_fwht_quant", "_fwht_quant_kernel")
    M(q + "indexer.weight_scale", "triton_poi_fused_mul_unsqueeze_0")
    M(q + "indexer.kpool_gate_score", NVJ, after=q + "indexer.weight_scale")
    M(q + "indexer.prefill_gather", GATHER, before_name="_kpool_softmax_rotate_write")
    M(q + "indexer.prefill_gather", GATHER, before=q + "indexer.prefill_gather")
    M(q + "indexer.kpool_decode_update", "_kpool_decode_update_batched_kernel")
    M(q + "indexer.kpool_prefill_write", "_kpool_softmax_rotate_write_cache_kernel")
    M(q + "indexer.kpool_tail_seed", "_kpool_tail_seed_kernel")
    M(q + "indexer.mqa_logits_decode", "sm90_fp8_paged_mqa_logits<")
    for n in ("FilteredTopKUnifiedKernel", "cooperative_topk_cs4", "cooperative_topk_cs2"):
        M(q + "indexer.topk_decode", n)
    M(q + "indexer.mqa_logits_prefill", "cp_gather_indexer_k_quant_cache_kernel")
    M(q + "indexer.mqa_logits_prefill", "sm90_fp8_mqa_logits<")
    M(q + "indexer.topk_prefill", "topKPerRowPrefill")
    M(q + "indexer.expand_pools", "_expand_pools_and_append_tail_kernel")
    M(q + "sparse_mla.mla_cache_append", "concat_and_cache_mla_kernel")
    M(q + "dense_prefill.kv_b_proj", NVJ, after=q + "sparse_mla.mla_cache_append",
      before_name="prepare_varlen_num_blocks")
    M(q + "dense_prefill.attention", "prepare_varlen_num_blocks", after=q + "dense_prefill.kv_b_proj")
    M(q + "dense_prefill.attention", "FlashAttnFwdSm90", after=q + "dense_prefill.attention")
    M(q + "q_absorb", NNT, after=q + "sparse_mla.mla_cache_append")
    M(q + "q_absorb", NNT, after=q + "dense_prefill.attention")
    M(q + "sparse_mla.index_remap", "sparse_mla_index_remap_kernel")
    M(q + "sparse_mla.q_rope_zero_fill", "FillFunctor<c10::BFloat16>", after=q + "sparse_mla.index_remap")
    M(q + "sparse_mla", "prepare_varlen_num_blocks", after=q + "sparse_mla.q_rope_zero_fill")
    M(q + "sparse_mla", "FlashAttnFwdSm90", after=q + "sparse_mla")
    M(q + "sparse_mla", "FlashAttnFwdCombine", after=q + "sparse_mla")
    M(q + "sparse_mla.pack_indices", "_pack_topk_indices", after=q + "sparse_mla.index_remap",
      source_experiment=FI_SRC)
    M(q + "sparse_mla", "BatchMLAPageAttentionHopperKernel", after=q + "sparse_mla.pack_indices",
      source_experiment=FI_SRC)
    # a prefill-only forward has no MQA rows: o_proj follows the dense call
    M(q + "o_proj", "_TNN", after=q + "dense_prefill.attention")
    M(q + "o_proj", "_TNT", after=q + "dense_prefill.attention")
    M(q + "v_up", NVJ, after=q + "sparse_mla")
    M(q + "o_proj", NVJ, after=q + "v_up")
    for g in ("q_b_proj", "indexer.wq_b", "indexer.wk_weights", "indexer.head_weights",
              "indexer.kpool_gate_score", "v_up", "o_proj"):
        M(q + g, SKR, after=q + g)
    glue_from = [q + s for s in (
        "indexer.kpool_gate_score", "indexer.prefill_gather", "indexer.kpool_prefill_write",
        "indexer.kpool_tail_seed", "indexer.kpool_decode_update", "indexer.mqa_logits_decode",
        "indexer.topk_decode", "indexer.mqa_logits_prefill", "indexer.topk_prefill",
        "indexer.expand_pools", "glue")]
    for prev in glue_from:
        for n in GLUE_NAMES:
            M(q + "glue", n, after=prev)


dsa_chain("")

# ======================================================================
# forward: dense FFN (3 layers) / MoE (42 layers)
# ======================================================================
F = "dense_ffn."
for s, role in [("gate_up_input_quant", "dense FFN gate/up input FP8 group quant"),
                ("gate_up", "dense FFN gate/up DeepGEMM"),
                ("act_and_mul", "dense FFN SiLU and multiply"),
                ("down_input_quant", "dense FFN down input FP8 group quant"),
                ("down", "dense FFN down DeepGEMM")]:
    op(F + s, role, [F + s])
S = "moe.shared_expert."
for s, role in [("gate_up_input_quant", "shared-expert gate/up input FP8 group quant"),
                ("gate_up", "shared-expert gate/up DeepGEMM"),
                ("act_and_mul", "shared-expert SiLU and multiply"),
                ("down_input_quant", "shared-expert down input FP8 group quant"),
                ("down", "shared-expert down DeepGEMM")]:
    op(S + s, role, [S + s])
R = "moe.routed."
op("moe.router.gate", "MoE router gate GEMM (fp32 out)", ["moe.router.gate"])
op("moe.router.topk", "MoE grouped top-k (single_group_topk)", ["moe.router.topk"])
for s, role in [("input_quant", "routed-expert FP8 input group quant"),
                ("align", "moe_align_block_size + count_and_sort_expert_tokens"),
                ("gate_up", "Triton FP8 fused_moe gate/up"),
                ("act_and_mul", "routed-expert SiLU and multiply"),
                ("down_quant", "routed-expert down input FP8 group quant"),
                ("down", "Triton FP8 fused_moe down"),
                ("sum", "moe_sum over top-k")]:
    op(R + s, role, [R + s])
op("moe.combine_glue", "shared + routed add", ["moe.combine_glue"])


def ffn_chain(prefix, first_evidence):
    for ev in first_evidence:
        M(prefix + "gate_up_input_quant", QUANT_DG, **ev)
    M(prefix + "gate_up", DG, after=prefix + "gate_up_input_quant")
    M(prefix + "act_and_mul", ACT, after=prefix + "gate_up")
    M(prefix + "down_input_quant", QUANT_DG, after=prefix + "act_and_mul")
    M(prefix + "down", DG, after=prefix + "down_input_quant")


def shared_backward(prefix, before_ops):
    """Folded bodies that open mid shared expert: anchor on what follows."""
    for b in before_ops:
        M(prefix + "down", DG, before=b)
    M(prefix + "down_input_quant", QUANT_DG, before=prefix + "down")
    M(prefix + "act_and_mul", ACT, before=prefix + "down_input_quant")
    M(prefix + "gate_up", DG, before=prefix + "act_and_mul")
    M(prefix + "gate_up_input_quant", QUANT_DG, before=prefix + "gate_up")


def routed_chain(P):
    """Router + Triton fused MoE + combine, forward or draft prefix."""
    M(P + "moe.router.gate", TSS)
    M(P + "moe.router.gate", "linearcute_dsl_ll_bf16")
    M(P + "moe.router.gate", SKR, after=P + "moe.router.gate")
    M(P + "moe.router.topk", "single_group_topk")
    M(P + R + "input_quant", QUANT_MOE, after=P + "moe.router.topk")
    M(P + R + "align", "moe_align_block_size_kernel")
    M(P + R + "align", "count_and_sort_expert_tokens_kernel")
    M(P + R + "gate_up", "fused_moe_kernel", after=P + R + "align")
    M(P + R + "gate_up", "fused_moe_kernel", before_name=ACT)
    M(P + R + "down", "fused_moe_kernel", before_name="moe_sum")
    M(P + R + "down_quant", QUANT_MOE, before=P + R + "down")
    M(P + R + "act_and_mul", ACT, after=P + R + "gate_up")
    M(P + R + "down_quant", QUANT_MOE, after=P + R + "act_and_mul")
    M(P + R + "down", "fused_moe_kernel", after=P + R + "down_quant")
    M(P + R + "sum", "moe_sum_vec_kernel")
    M(P + "moe.combine_glue", "CUDAFunctor_add<c10::BFloat16>", after=P + R + "sum")
    M(P + "moe.combine_glue", "CUDAFunctor_add<c10::BFloat16>", after=P + S + "down")


ffn_chain(F, [dict(after_name=BIGFUSE)])
# shared expert: after the router GEMM (inline, > 256 tokens) or opening the
# main-stream body (aux-stream overlap, <= 256 tokens)
ffn_chain(S, [dict(after="moe.router.gate"), dict(after_name="<none>")])
shared_backward(S, ["moe.combine_glue", "moe.router.topk"])
# the last MoE layer's add opens the main-stream tail run (aux-stream decode)
M("moe.combine_glue", "CUDAFunctor_add<c10::BFloat16>", after_name="<none>", before="tp.all_reduce")
M("moe.combine_glue", "CUDAFunctor_add<c10::BFloat16>", after_name=DG, before="tp.all_reduce")
routed_chain("")

# ======================================================================
# draft: MTP layer (step 0 + K-1 index-sharing steps)
# ======================================================================
for s, role in [
        ("step_glue", "draft step input prep and attention-metadata rebuild "
                      "(slot mappings, token/request maps, paged-MQA metadata, draft inputs)"),
        ("embedding", "MTP vocab-parallel embedding"),
        ("eh_norm", "MTP fused enorm/hnorm"),
        ("eh_proj", "MTP eh projection"),
        ("input_norm", "MTP decoder input RMSNorm"),
        ("post_attn_norm", "MTP post-attention fused add + RMSNorm"),
        ("head_norm", "MTP shared-head fused add + RMSNorm"),
        ("row_select", "MTP last-token row select before the head"),
        ("lm_head", "MTP shared language-model head"),
        ("argmax", "MTP greedy argmax over the gathered logits")]:
    op("mtp." + s, role, [s])
op("mtp.tp.all_reduce", "MTP TP all-reduce after embedding, attention and FFN",
   ["embedding_all_reduce.fusion", "embedding_all_reduce.fallback",
    "attn_all_reduce.fusion", "attn_all_reduce.fallback",
    "ffn_all_reduce.fusion", "ffn_all_reduce.fallback"], "collective")
op("mtp.logits_gather", "MTP vocab-parallel logits all-gather and its concat copy", ["logits_gather"], "collective")
for name, (role, slots, typ) in list(ROLE.items()):
    if name.startswith(("dsa.", "moe.")):
        op("mtp." + name, "MTP " + role, slots, typ)

M("mtp.embedding", "vocab_parallel_embedding_kernel")
for n in ALL_REDUCES:
    M("mtp.tp.all_reduce", n)
M("mtp.eh_norm", "_fused_eh_norm_kernel")
M("mtp.eh_proj", NVJ, after="mtp.eh_norm")
M("mtp.eh_proj", SKR, after="mtp.eh_proj")
M("mtp.input_norm", "vllm::rms_norm_kernel")
M("mtp.dsa.fused_qkv_a", NVJ, after="mtp.input_norm")
dsa_chain("mtp.")
M("mtp.post_attn_norm", FADD_NORM, before_name=QUANT_DG)
M("mtp.post_attn_norm", FADD_NORM, before_name=TSS)
M("mtp.head_norm", FADD_NORM, before_name=GATHER)
M("mtp.head_norm", FADD_NORM, before_name=NVJ)
ffn_chain("mtp." + S, [dict(after="mtp.post_attn_norm"), dict(after="mtp.moe.router.gate")])
routed_chain("mtp.")
M("mtp.row_select", GATHER, after="mtp.head_norm")
M("mtp.lm_head", NVJ, before_name="ncclDevKernel_AllGather")
M("mtp.logits_gather", "ncclDevKernel_AllGather_RING_LL")
M("mtp.logits_gather", COPY_BF16, after="mtp.logits_gather")
M("mtp.argmax", "ArgMaxOps")
for n in ("_compute_slot_mappings_kernel", "_token_request_mapping_kernel", "sm90_paged_mqa_logits_metadata",
          "_update_draft_inputs_kernel", "_prepare_decode_inputs_kernel", "_prepare_prefill_inputs_kernel"):
    M("mtp.step_glue", n)
# FLASHINFER_MLA_SPARSE_SM90's per-plan kv_end clamp, part of the draft metadata rebuild; the
# forward plan's copy runs in preprocess and stays unmapped there.
M("mtp.step_glue", "_clamp_work_kv_end", source_experiment=FI_SRC)
STEP_GLUE_NAMES = GLUE_NAMES + (IDX, GATHER, "vectorized_gather_kernel<(int)16, int>", "DeviceScanInitKernel",
                                 "DeviceScanKernel", "compute_cuda_kernel", "_scatter_gather_elementwise_kernel",
                                 "indexSelectSmallIndex<int")
for n in STEP_GLUE_NAMES:
    M("mtp.step_glue", n, after="mtp.step_glue")
    M("mtp.step_glue", n, before="mtp.step_glue")
M("mtp.step_glue", "kernel", name_exact=True, after="mtp.step_glue")

json.dump({"schema_version": 1,
           "note": "GLM-5.3-Flash FP8 vLLM v0.31 TP4/DP1 (no EP) MTP-5 on H200 "
                   "(glm53_flash_vllm_fp8_kda_dsa_moe_speculative). One order-free rule set; generated by "
                   "gen_rules.py in this directory, then pruned to rules that match at the fixpoint.",
           "rules": rules}, open(sys.argv[1], "w"), indent=1)
print(len(rules), "rules")
