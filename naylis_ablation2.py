print("v2")

"""
Naylis Ablation - Graph MoE Edition

Remplacement de PKM par Graph-MoE block-sparse.
Variants:
  vanilla
  vanilla_thin_ffn
  naylis_global
  naylis_layeroff
  naylis_graph_moe
  naylis_local

Corrections importantes:
  - PKM retire car trop couteux.
  - Graph-MoE block-sparse ajoute.
  - Correction du bug de loss initiale > log(V):
    reinitialisation controlee de l'embedding d'entree/sortie.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download, HfApi
from transformers import (
    LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers import apply_liger_kernel_to_llama, LigerRMSNorm

import matplotlib
matplotlib.use("Agg")  # pas d'affichage graphique sur Modal, juste sauvegarde en PNG
import matplotlib.pyplot as plt

# ============================================================
# CONFIG COMMUNE
# ============================================================
SEED = 257
TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
DATA_HF_REPO_ID = "TheRealSkyline/5B_Tokens_Cosmopedia-V2"
DATA_HF_FILENAME = "pretrain_data_5B.bin"
DATA_HF_REPO_TYPE = "dataset"
SEQ_LEN = 1024
VOCAB_SIZE = 49152
D_MODEL = 1024
N_LAYERS = 20
N_HEADS = 16

# Memoire globale dense / Graph-MoE
MEM_SIZE = 1024
GRAPH_RATIO = 3

# Graph-MoE
GRAPH_MOE_BLOCK_SIZE = 128
GRAPH_MOE_TOP_BLOCKS = 2       # 2 * 128 = 256 slots actifs
GRAPH_MOE_ROUTER_RANK = 32
GRAPH_MOE_ROUTE_LEVEL = "sequence"  # "sequence" ou "batch"
MOE_AUX_WEIGHT = 0.01

DIAG_EVERY_N_CALLS = 500
TOTAL_TOKENS = 5_000_000_000

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN n'est pas défini. Lance avec: "
        "HF_TOKEN=hf_xxx python naylis_ablation2.py ..."
    )
HF_REPO_ID = "TheRealSkyline/naylis_ablation_300M"
HF_REPO_TYPE = "dataset"
CHECKPOINT_INTERVAL_SEC = 3600
PEAK_FLOPS = 125e12  # A10G bf16

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# MODULES GRAPH MEMORY
# ============================================================
class GraphMemoryDense(nn.Module):
    """
    naylis_global / naylis_layeroff / naylis_local:
    attention softmax dense classique vers M.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.graph_scale = nn.Parameter(torch.zeros(1))

        self.last_entropy = None
        self._call_count = 0

    def forward(self, x, M):
        B, S, D = x.shape

        # Cast memoire au dtype de x pour eviter les mismatches bf16/fp32
        M = M.to(x.dtype)

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(M).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)
        v = self.v_proj(M).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        self._call_count += 1
        if self._call_count % DIAG_EVERY_N_CALLS == 0:
            with torch.no_grad():
                scale = 1.0 / (self.head_dim ** 0.5)
                scores = torch.einsum("bhsd,bhmd->bhsm", q, k) * scale
                probs = F.softmax(scores.float(), dim=-1)
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                self.last_entropy = entropy.item()

        return self.graph_scale * self.o_proj(out)


class GraphMemoryMoE(nn.Module):
    """
    Graph-MoE block-sparse.

    La memoire globale M est representee sous forme de blocs:
      M_blocks: [num_blocks, block_size * d_model]

    Le routeur selectionne top_blocks blocs.
    Chaque bloc contient block_size slots.
    L'attention est faite seulement sur les slots actifs.

    Exemple:
      MEM_SIZE = 1024
      block_size = 128
      top_blocks = 2
      -> 256 slots actifs
    """
    def __init__(
        self,
        d_model,
        n_heads,
        mem_size,
        block_size,
        top_blocks,
        router_rank=32,
        route_level="sequence",
    ):
        super().__init__()

        assert mem_size % block_size == 0, "MEM_SIZE doit etre divisible par GRAPH_MOE_BLOCK_SIZE"

        self.num_blocks = mem_size // block_size
        self.block_size = block_size
        self.top_blocks = top_blocks
        self.active_slots = top_blocks * block_size

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.route_level = route_level

        # Router low-rank
        self.router_down = nn.Linear(d_model, router_rank, bias=False)
        self.router_up = nn.Linear(router_rank, self.num_blocks, bias=False)

        # Projections graph
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.graph_scale = nn.Parameter(torch.zeros(1))

        # Diagnostics
        self.register_buffer("block_usage", torch.zeros(self.num_blocks, dtype=torch.long), persistent=False)
        self.last_entropy = None
        self.last_dead_frac = None

        # Pour auxiliary load-balancing loss
        self.last_router_probs = None
        self.last_top_idx = None

        # noise_scale: controle l'amplitude du bruit Gumbel pendant le
        # training. Mis a 0 apres warmup par NoiseAnnealCallback pour que
        # le routing d'entrainement finisse par matcher le routing
        # deterministe utilise en eval (sinon: routes aleatoires a
        # l'entrainement, routes figees a l'eval -> mismatch train/eval).
        self.noise_scale = 1.0

        self._call_count = 0

    def forward(self, x, M_blocks):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------
        if self.route_level == "batch":
            h = x.mean(dim=(0, 1))  # [D]
            router_logits = self.router_up(torch.tanh(self.router_down(h)))  # [num_blocks]
            router_logits = router_logits.unsqueeze(0)  # [1, num_blocks]
        else:
            h = x.mean(dim=1)  # [B, D]
            router_logits = self.router_up(torch.tanh(self.router_down(h)))  # [B, num_blocks]

        router_logits = router_logits.float()

        # Gumbel noise pendant l'entrainement pour exploration.
        # noise_scale est mis a 0 apres warmup (voir NoiseAnnealCallback).
        if self.training and self.noise_scale > 0:
            noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-9) + 1e-9)
            noisy_logits = router_logits + self.noise_scale * noise
        else:
            noisy_logits = router_logits

        router_probs = F.softmax(router_logits, dim=-1)
        top_values, top_idx = noisy_logits.topk(self.top_blocks, dim=-1)  # [R, top_blocks], R=B ou 1

        # Gate differentiable: top_values vient directement de router_logits
        # (topk garde le gradient sur les VALEURS selectionnees, meme si les
        # INDICES restent non-differentiables). En ponderant la contribution
        # de chaque bloc par ce gate, la loss LM peut enfin remonter jusqu'au
        # routeur: si le bloc choisi aide a reduire la loss, le gradient
        # augmente son gate pour des entrees similaires. Sans ca, seule la
        # loss d'equilibrage (load-balancing) touchait le routeur.
        top_gates = F.softmax(top_values, dim=-1)  # [R, top_blocks]

        # ----------------------------------------------------
        # Gather des blocs actifs
        # ----------------------------------------------------
        if self.route_level == "batch":
            active_flat = F.embedding(top_idx.squeeze(0), M_blocks)  # [top_blocks, block_size*D]
            active = active_flat.view(self.active_slots, D).to(q.dtype)  # [active_slots, D]

            k = self.k_proj(active).view(1, self.active_slots, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(active).view(1, self.active_slots, self.n_heads, self.head_dim).transpose(1, 2)

            k = k.expand(B, -1, -1, -1)
            v = v.expand(B, -1, -1, -1)

            gates_per_slot = top_gates.squeeze(0)  # [top_blocks]
        else:
            active_flat = F.embedding(top_idx, M_blocks)  # [B, top_blocks, block_size*D]
            active = active_flat.view(B, self.active_slots, D).to(q.dtype)

            k = self.k_proj(active).view(B, self.active_slots, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(active).view(B, self.active_slots, self.n_heads, self.head_dim).transpose(1, 2)

            gates_per_slot = top_gates  # [B, top_blocks]

        # log(gate) repete sur les block_size slots de chaque bloc, ajoute
        # comme biais additif aux scores d'attention avant le softmax
        # (SDPA additionne attn_mask flottant aux scores QK^T*scale).
        # C'est le mecanisme standard de ponderation top-k des MoE.
        gate_bias = gates_per_slot.clamp_min(1e-9).log()  # [.., top_blocks]
        gate_bias = gate_bias.unsqueeze(-1).expand(*gate_bias.shape, self.block_size)
        gate_bias = gate_bias.reshape(*gate_bias.shape[:-2], self.active_slots)  # [.., active_slots]
        gate_bias = gate_bias.reshape(-1, 1, 1, self.active_slots).to(q.dtype)  # broadcast heads + positions

        # ----------------------------------------------------
        # Attention sur slots actifs (biaisee par le gate du routeur)
        # ----------------------------------------------------
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=gate_bias, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Sauvegarde pour auxiliary loss
        if self.training:
            self.last_router_probs = router_probs
            self.last_top_idx = top_idx

        # Usage tracking
        with torch.no_grad():
            self.block_usage.scatter_add_(
                0,
                top_idx.reshape(-1),
                torch.ones(top_idx.numel(), dtype=torch.long, device=top_idx.device),
            )

        self._call_count += 1
        if self._call_count % DIAG_EVERY_N_CALLS == 0:
            with torch.no_grad():
                entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(-1).mean()
                self.last_entropy = entropy.item()
                self.last_dead_frac = (self.block_usage == 0).float().mean().item()
                self.block_usage.zero_()

        return self.graph_scale * self.o_proj(out)


# ============================================================
# NAYLIS LAYER / MODEL
# ============================================================
class NaylisDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, variant, mem_size, graph_ratio, local_M=False):
        super().__init__(config, layer_idx)
        self.variant = variant
        self.local_M = local_M

        self.has_graph = True
        if variant == "naylis_layeroff":
            self.has_graph = (layer_idx % graph_ratio == 0)

        if self.has_graph:
            self.graph_norm = LigerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            if variant == "naylis_graph_moe":
                self.graph_mem = GraphMemoryMoE(
                    d_model=config.hidden_size,
                    n_heads=config.num_attention_heads,
                    mem_size=mem_size,
                    block_size=GRAPH_MOE_BLOCK_SIZE,
                    top_blocks=GRAPH_MOE_TOP_BLOCKS,
                    router_rank=GRAPH_MOE_ROUTER_RANK,
                    route_level=GRAPH_MOE_ROUTE_LEVEL,
                )
            else:
                self.graph_mem = GraphMemoryDense(config.hidden_size, config.num_attention_heads)

            if local_M:
                self.M_local = nn.Parameter(torch.randn(mem_size, config.hidden_size) * 0.02)

    def forward(self, hidden_states, M=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states=hidden_states, position_embeddings=position_embeddings, **kwargs)
        hidden_states = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
        hidden_states = residual + hidden_states

        if self.has_graph:
            residual = hidden_states
            hidden_states = self.graph_norm(hidden_states)

            mem = self.M_local if self.local_M else M
            hidden_states = self.graph_mem(hidden_states, mem)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class NaylisLlamaModel(LlamaModel):
    def __init__(self, config, variant, mem_size, graph_ratio):
        super().__init__(config)

        local_M = (variant == "naylis_local")

        self.has_global_M = variant in ("naylis_global", "naylis_layeroff")
        self.has_global_M_blocks = variant == "naylis_graph_moe"

        if self.has_global_M:
            self.M = nn.Parameter(torch.randn(mem_size, config.hidden_size) * 0.02)

        if self.has_global_M_blocks:
            assert mem_size % GRAPH_MOE_BLOCK_SIZE == 0
            num_blocks = mem_size // GRAPH_MOE_BLOCK_SIZE
            self.M_blocks = nn.Parameter(
                torch.randn(num_blocks, GRAPH_MOE_BLOCK_SIZE * config.hidden_size) * 0.02
            )

        self.layers = nn.ModuleList([
            NaylisDecoderLayer(
                config,
                layer_idx=i,
                variant=variant,
                mem_size=mem_size,
                graph_ratio=graph_ratio,
                local_M=local_M,
            )
            for i in range(config.num_hidden_layers)
        ])

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        M = None
        if self.has_global_M:
            M = self.M
        elif self.has_global_M_blocks:
            M = self.M_blocks

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                M=M,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs
            )[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class NaylisLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, variant, mem_size=MEM_SIZE, graph_ratio=GRAPH_RATIO):
        super().__init__(config)

        self.model = NaylisLlamaModel(config, variant, mem_size, graph_ratio)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # reduction="sum" : on divise nous-memes par num_items_in_batch (voir forward),
        # sinon Trainer et le module double la normalisation en grad accumulation.
        self.liger_lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        self.moe_aux_weight = MOE_AUX_WEIGHT

        self.post_init()

    def _collect_moe_aux_loss(self, device):
        """
        Load-balancing loss inspiree des MoE:
          aux = num_blocks * sum_i(frac_selected_i * mean_prob_i)
        """
        aux = torch.zeros((), device=device, dtype=torch.float32)
        count = 0

        for module in self.modules():
            if isinstance(module, GraphMemoryMoE):
                if module.last_router_probs is None or module.last_top_idx is None:
                    continue

                probs = module.last_router_probs.float()  # [R, num_blocks]
                top_idx = module.last_top_idx             # [R, top_blocks]

                selected = torch.zeros_like(probs)
                selected.scatter_(1, top_idx, 1.0)

                frac_selected = selected.mean(dim=0)
                mean_prob = probs.mean(dim=0)

                aux = aux + float(module.num_blocks) * (frac_selected * mean_prob).sum()

                # On libere le graphe intermediaire
                module.last_router_probs = None
                module.last_top_idx = None

                count += 1

        if count > 0:
            aux = aux / count

        return aux

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
                num_items_in_batch=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state

        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)

            # reduction="sum" (voir __init__) : on normalise nous-memes.
            # - en entrainement, Trainer fournit num_items_in_batch = nb total de
            #   tokens cibles sur TOUT le batch accumule (somme sur les grad_accum
            #   micro-batches). Diviser par cette valeur donne la vraie moyenne du
            #   macro-batch, et Trainer ne divise plus par gradient_accumulation_steps
            #   pour les modeles qui exposent **kwargs (cf. num_items_in_batch dans
            #   la signature) => il ne faut donc PAS diviser une seconde fois par
            #   grad_accum, sinon la loss/gradient est trop petite; et il ne faut pas
            #   non plus la laisser en simple moyenne par micro-batch, sinon c'est
            #   l'ancien bug qui la double (Trainer ne redivise pas plus).
            # - hors entrainement (eval/generate), num_items_in_batch est absent:
            #   on retombe sur une moyenne classique sur le micro-batch courant.
            loss = self.liger_lce(
                self.lm_head.weight,
                shift_hidden.view(-1, shift_hidden.size(-1)),
                shift_labels,
            )
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch
            else:
                n_valid = (shift_labels != -100).sum().clamp_min(1)
                loss = loss / n_valid

            if self.training:
                aux_loss = self._collect_moe_aux_loss(hidden_states.device)
                loss = loss + self.moe_aux_weight * aux_loss

            logits = None
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        return CausalLMOutputWithPast(loss=loss, logits=logits)


# ============================================================
# FIX INITIALISATION LOSS
# ============================================================
def stabilize_output_embedding_init(model, config, seed):
    """
    Correction du bug de loss initiale > log(V).

    Cause principale dans les classes custom heritees de Llama:
      - double construction partielle du modele via super().__init__
      - consommation du RNG PyTorch avant l'initialisation finale
      - lm_head / embed_tokens peuvent se retrouver avec une init defavorable

    Cette fonction reinitialise proprement l'embedding d'entree/sortie
    avec un seed dedie, ce qui stabilise la loss initiale autour de log(V).
    """
    torch.manual_seed(seed + 99991)

    std = config.initializer_range

    emb = model.get_input_embeddings()
    if emb is not None:
        emb.weight.data.normal_(mean=0.0, std=std)

    if config.tie_word_embeddings:
        if hasattr(model, "lm_head"):
            model.lm_head.weight = emb.weight
    else:
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            model.lm_head.weight.data.normal_(mean=0.0, std=std)

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        if hasattr(model.model.norm, "weight"):
            model.model.norm.weight.data.fill_(1.0)

    # Restore main seed
    torch.manual_seed(seed)


# ============================================================
# DATASET
# ============================================================
class BinDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, seq_len, total_tokens_to_read=None):
        self.seq_len = seq_len
        self.tokens = np.memmap(file_path, dtype=np.uint16, mode="r")
        if total_tokens_to_read:
            self.tokens = self.tokens[:total_tokens_to_read]
        self.n_examples = len(self.tokens) // (seq_len + 1)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1].astype(np.int64)
        # input_ids[t] = token[t], labels[t] = token[t] (pas de pre-shift ici).
        # Le shift interne au forward (shift_hidden vs shift_labels[..., 1:])
        # se charge de produire hidden[t] -> labels[t+1] = token[t+1].
        input_ids = torch.tensor(chunk[:-1])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# ============================================================
# CALLBACK MFU
# ============================================================
def estimate_training_flops_per_token(variant, n_params, micro_batch_size):
    """
    Estimation des FLOPs d'entrainement par token cible (forward + backward).

    La base 6 * N est correcte pour les poids denses executes a chaque token,
    lm_head inclus (le poids tied embed/lm_head est compte une seule fois dans
    n_params mais est bien utilise par la cross-entropy). Les branches graphe
    demandent une correction : q/o sont token-wise, tandis que k/v, le routeur
    et la memoire sont calcules par sequence ou par batch. On ajoute egalement
    le cout QK + AV de la cross-attention, absent de 6 * N.

    Cela reste une estimation analytique : tokens/s et VRAM restent les mesures
    systeme de reference, tandis que ce MFU sert a comparer les variantes avec
    la meme convention de calcul.
    """
    # Poids denses appliques a chaque token, plus le cout supplementaire de
    # l'attention causale standard (QK et AV) dans tous les layers.
    flops = 6 * n_params + 12 * N_LAYERS * D_MODEL * SEQ_LEN

    graph_variants = {
        "naylis_global",
        "naylis_layeroff",
        "naylis_graph_moe",
        "naylis_local",
    }
    if variant not in graph_variants:
        return flops

    graph_layers = N_LAYERS
    if variant == "naylis_layeroff":
        graph_layers = len(range(0, N_LAYERS, GRAPH_RATIO))

    if variant == "naylis_graph_moe":
        active_slots = GRAPH_MOE_TOP_BLOCKS * GRAPH_MOE_BLOCK_SIZE
        num_blocks = MEM_SIZE // GRAPH_MOE_BLOCK_SIZE
        router_params = D_MODEL * GRAPH_MOE_ROUTER_RANK + GRAPH_MOE_ROUTER_RANK * num_blocks
    else:
        active_slots = MEM_SIZE
        router_params = 0

    # Dans le code, q/o sont appliques a chaque token. k/v et le routeur sont
    # appliques une fois par sequence (ou une fois par micro-batch), pas une
    # fois par token. La formule 6*N les comptait donc a tort comme token-wise.
    if variant == "naylis_graph_moe" and GRAPH_MOE_ROUTE_LEVEL == "batch":
        units_per_token = 1.0 / (micro_batch_size * SEQ_LEN)
    else:
        units_per_token = 1.0 / SEQ_LEN

    graph_proj_params = 4 * D_MODEL * D_MODEL  # q, k, v, o dans chaque graph layer
    incorrectly_tokenwise = graph_layers * (graph_proj_params + router_params)

    # M est une table de slots, pas une matrice appliquee directement a chaque
    # token. Le cout reel vient des projections k/v calculees sur les slots.
    if variant == "naylis_local":
        memory_params = graph_layers * MEM_SIZE * D_MODEL
    else:
        memory_params = MEM_SIZE * D_MODEL

    flops -= 6 * (incorrectly_tokenwise + memory_params)

    q_o_flops = 6 * (2 * D_MODEL * D_MODEL)
    k_v_flops = 6 * (2 * D_MODEL * D_MODEL) * active_slots * units_per_token
    router_flops = 6 * router_params * units_per_token
    cross_attention_flops = 12 * D_MODEL * active_slots  # QK + AV, forward + backward

    flops += graph_layers * (q_o_flops + k_v_flops + router_flops + cross_attention_flops)
    return flops


class MFUCallback(TrainerCallback):
    def __init__(self, tokens_per_step, flops_per_token, peak_flops, total_tokens, model=None):
        self.tokens_per_step = tokens_per_step
        self.flops_per_token = flops_per_token
        self.peak_flops = peak_flops
        self.total_tokens = total_tokens
        self.model = model
        self.t0 = None
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        # IMPORTANT: state.global_step est le compteur ABSOLU (recharge
        # depuis le checkpoint apres une reprise, ex: ~25000, pas 0).
        # Sans cet offset, tok_per_sec = tokens_per_step*global_step/elapsed
        # utilise le travail CUMULE depuis le tout debut au numerateur mais
        # seulement le temps de CETTE session au denominateur -> Tok/s et
        # MFU enormement gonfles juste apres une reprise, qui redescendent
        # au fil des minutes vers la vraie valeur (exactement ce qui a ete
        # observe: ETA=2min qui "remonte" progressivement vers la realite).
        self.step0 = state.global_step

    def _graph_scale_stats(self):
        vals = [p.abs().item() for n, p in self.model.named_parameters() if n.endswith("graph_scale")]
        return (sum(vals) / len(vals), max(vals)) if vals else None

    def _memory_norm_stats(self):
        norms = {}

        for n, p in self.model.named_parameters():
            if n.endswith(".M") or n == "model.M":
                norms.setdefault("M", []).append(p.norm().item())
            elif n.endswith("M_local"):
                norms.setdefault("M_local(avg/max)", []).append(p.norm().item())
            elif n.endswith("M_blocks") or n == "model.M_blocks":
                norms.setdefault("M_blocks", []).append(p.norm().item())

        parts = []
        for label, vs in norms.items():
            if len(vs) == 1:
                parts.append(f"{label}={vs[0]:.3f}")
            else:
                parts.append(f"{label}={sum(vs)/len(vs):.3f}/{max(vs):.3f}")

        return ", ".join(parts) if parts else None

    def _diag_stats(self):
        entropies = [
            m.last_entropy for m in self.model.modules()
            if isinstance(m, (GraphMemoryDense, GraphMemoryMoE)) and m.last_entropy is not None
        ]

        dead_fracs = [
            m.last_dead_frac for m in self.model.modules()
            if isinstance(m, GraphMemoryMoE) and m.last_dead_frac is not None
        ]

        parts = []
        if entropies:
            parts.append(f"entropie_attn(avg)={sum(entropies)/len(entropies):.3f}")
        if dead_fracs:
            parts.append(f"blocks_morts={sum(dead_fracs)/len(dead_fracs)*100:.1f}%")

        return ", ".join(parts) if parts else None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs or self.t0 is None:
            return

        elapsed = time.time() - self.t0
        if elapsed <= 0:
            return

        # steps_this_session (pas state.global_step tout seul!) pour que le
        # debit ne compte que le travail reellement fait DANS cette session
        # -> reste stable et correct des le premier log apres une reprise.
        steps_this_session = max(state.global_step - self.step0, 0)
        tok_per_sec = (self.tokens_per_step * steps_this_session) / elapsed
        mfu = (self.flops_per_token * tok_per_sec) / self.peak_flops
        # tokens_done/remaining/ETA restent bases sur state.global_step
        # ABSOLU: c'est la progression totale vers l'objectif qui compte ici,
        # pas seulement cette session.
        tokens_done = self.tokens_per_step * state.global_step
        remaining = max(self.total_tokens - tokens_done, 0)
        eta_min = (remaining / tok_per_sec) / 60 if tok_per_sec > 0 else float("inf")

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")

        pieces = [
            f"loss={loss:.4f}",
            f"grad_norm={grad_norm:.2f}",
            f"lr={lr:.2e}",
            f"Tok/s={tok_per_sec:,.0f}",
            f"MFU(est.)={mfu*100:.1f}%",
            f"Tokens={tokens_done/1e6:.0f}M/{self.total_tokens/1e6:.0f}M",
            f"ETA={eta_min:.0f}min",
        ]

        gs = self._graph_scale_stats()
        if gs:
            pieces.append(f"graph_scale(avg/max)={gs[0]:.4f}/{gs[1]:.4f}")

        mem_norm = self._memory_norm_stats()
        if mem_norm:
            pieces.append(mem_norm)

        diag = self._diag_stats()
        if diag:
            pieces.append(diag)

        msg = " | ".join(pieces)

        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str(msg)
        else:
            tqdm.write(f"[Step {state.global_step}] {msg}")


class NaylisTrainer(Trainer):
    """
    Trainer.log() ecrit deja dans state.log_history AVANT que les callbacks
    on_log ne soient appeles -> un TrainerCallback classique ne peut pas
    ajouter de cles au log persiste. Il faut intercepter log() lui-meme.
    """
    def log(self, logs, *args, **kwargs):
        logs.update(self._collect_moe_router_stats())
        super().log(logs, *args, **kwargs)

    def _collect_moe_router_stats(self):
        entropies, dead_fracs = [], []
        for module in self.model.modules():
            e = getattr(module, "last_entropy", None)
            d = getattr(module, "last_dead_frac", None)
            if e is not None:
                try: entropies.append(float(e))
                except (TypeError, ValueError): pass
            if d is not None:
                try: dead_fracs.append(float(d))
                except (TypeError, ValueError): pass
        stats = {}
        if entropies:
            stats["moe_entropy_mean"] = sum(entropies) / len(entropies)
            stats["moe_entropy_min"] = min(entropies)
        if dead_fracs:
            stats["moe_dead_frac_mean"] = sum(dead_fracs) / len(dead_fracs)
            stats["moe_dead_frac_max"] = max(dead_fracs)
        return stats


def merge_log_histories(*histories):
    """
    Fusionne plusieurs log_history (listes de dicts avec une cle 'step')
    en dedupliquant par step. En cas de step present dans plusieurs listes,
    la DERNIERE liste passee en argument l'emporte (on l'appelle donc avec
    la liste la plus recente/complete en dernier).
    """
    by_step = {}
    for history in histories:
        for entry in history or []:
            step = entry.get("step")
            if step is not None:
                by_step[step] = entry
    return [by_step[s] for s in sorted(by_step)]


def load_log_history_from_checkpoint(checkpoint_dir):
    """Lit trainer_state.json d'un dossier de checkpoint local. Renvoie []
    si absent (ex: pas de reprise, premier run)."""
    if not checkpoint_dir:
        return []
    path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("log_history", [])
    except Exception as e:
        print(f"[Curves] Impossible de lire {path}: {e}")
        return []


def generate_training_curves(log_history, output_dir, run_label):
    """
    Un PNG par metrique numerique presente dans log_history, sauf ppl/perplexity.
    """
    os.makedirs(output_dir, exist_ok=True)
    series = defaultdict(list)
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for key, value in entry.items():
            if key in ("step", "epoch"):
                continue
            if "ppl" in key.lower() or "perplexity" in key.lower():
                continue
            if isinstance(value, (int, float)):
                series[key].append((step, value))

    saved = []
    for metric_name, points in series.items():
        if not points:
            continue
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        plt.figure(figsize=(9, 5))
        plt.plot(xs, ys, linewidth=1.2)
        plt.xlabel("step")
        plt.ylabel(metric_name)
        plt.title(f"{run_label} - {metric_name}")
        plt.grid(alpha=0.3)
        out_path = os.path.join(output_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close()
        saved.append(out_path)
        print(f"[Curves]   -> {out_path}")

    return saved


def push_curves_to_hub(curves_dir, variant, seed):
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True)
        api.upload_folder(
            folder_path=curves_dir,
            path_in_repo=f"{variant}_seed{seed}/curves",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"[Curves] Poussées vers {HF_REPO_ID}/{variant}_seed{seed}/curves")
    except Exception as e:
        print(f"[Curves] ATTENTION échec upload des courbes: {e}")


class NoiseAnnealCallback(TrainerCallback):
    """
    Coupe le bruit Gumbel des modules GraphMemoryMoE apres cutoff_frac de
    l'entrainement. Objectif: le routing dur pendant les derniers steps de
    training devient deterministe, comme en eval - on evite d'optimiser
    les poids (q/k/v/o_proj des blocs, M_blocks) sur une distribution de
    choix de blocs qui ne correspondra jamais a ce qui sera reellement
    utilise a l'inference.
    """
    def __init__(self, model, cutoff_frac=0.15):
        self.model = model
        self.cutoff_frac = cutoff_frac
        self._disabled = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._disabled or state.max_steps <= 0:
            return control
        if state.global_step >= self.cutoff_frac * state.max_steps:
            for module in self.model.modules():
                if hasattr(module, "noise_scale"):
                    module.noise_scale = 0.0
            self._disabled = True
            print(
                f"\n[NoiseAnneal] Bruit Gumbel désactivé au step {state.global_step} "
                f"(cutoff={self.cutoff_frac:.0%} de {state.max_steps} steps)"
            )
        return control


class HourlyCheckpointCallback(TrainerCallback):
    def __init__(self, variant, seed, interval_sec=3600):
        self.variant = variant
        self.seed = seed
        self.interval_sec = interval_sec
        self.last_save = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_save = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.last_save >= self.interval_sec:
            control.should_save = True
            self.last_save = time.time()
        return control

    def on_save(self, args, state, control, **kwargs):
        import glob
        ckpts = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1])
        )
        if not ckpts:
            return control

        latest = ckpts[-1]
        try:
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True)
            api.upload_folder(
                folder_path=latest,
                path_in_repo=f"{self.variant}_seed{self.seed}/checkpoint",
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
            )
            print(f"\n[Checkpoint] step {state.global_step} poussé vers {HF_REPO_ID}/{self.variant}_seed{self.seed}/checkpoint")
        except Exception as e:
            print(f"\n[Checkpoint] ATTENTION échec upload: {e}")

        return control


def try_download_checkpoint(variant, seed):
    from huggingface_hub import snapshot_download
    key = f"{variant}_seed{seed}"

    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
            allow_patterns=f"{key}/checkpoint/*",
        )
        resume_path = os.path.join(local_dir, key, "checkpoint")
        if os.path.isdir(resume_path) and os.listdir(resume_path):
            print(f"[Resume] Checkpoint trouvé: {resume_path}")
            return resume_path
    except Exception as e:
        print(f"[Resume] Aucun checkpoint exploitable sur le Hub ({e}), démarrage à zéro.")

    return None


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "vanilla",
            "vanilla_thin_ffn",
            "naylis_global",
            "naylis_layeroff",
            "naylis_graph_moe",
            "naylis_local",
        ]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"

    print(f"Téléchargement de {DATA_HF_FILENAME}...")
    data_path = hf_hub_download(
        repo_id=DATA_HF_REPO_ID,
        filename=DATA_HF_FILENAME,
        repo_type=DATA_HF_REPO_TYPE,
    )
    train_dataset = BinDataset(data_path, SEQ_LEN, total_tokens_to_read=TOTAL_TOKENS)

    if args.variant in ("vanilla", "vanilla_thin_ffn"):
        apply_liger_kernel_to_llama()
        if args.variant == "vanilla":
            intermediate_size = int(8 / 3 * D_MODEL / 64) * 64
        else:
            # vanilla_thin_ffn: meme Llama standard (pas de graph memory),
            # mais ffn = d_model au lieu de ~2.67x d_model, pour isoler
            # l'effet de la largeur du FFN independamment des ajouts Naylis.
            intermediate_size = D_MODEL

        config = LlamaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=D_MODEL,
            intermediate_size=intermediate_size,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_HEADS,
            max_position_embeddings=SEQ_LEN,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attention_bias=False,
            tie_word_embeddings=True,
            attn_implementation="sdpa",
            use_cache=False,
        )
        model = LlamaForCausalLM(config)
    else:
        apply_liger_kernel_to_llama()
        intermediate_size = D_MODEL

        config = LlamaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=D_MODEL,
            intermediate_size=intermediate_size,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_HEADS,
            max_position_embeddings=SEQ_LEN,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attention_bias=False,
            tie_word_embeddings=True,
            attn_implementation="sdpa",
            use_cache=False,
        )
        model = NaylisLlamaForCausalLM(config, variant=args.variant)

    # Correction importante: stabilise la loss initiale autour de log(V)
    stabilize_output_embedding_init(model, config, args.seed)

    # Restore seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Variant: {args.variant} | Params: {n_params/1e6:.1f}M | intermediate_size={intermediate_size}")

    flops_per_token = estimate_training_flops_per_token(
        args.variant,
        n_params,
        args.batch_size,
    )
    print(f"MFU FLOPs estimate: {flops_per_token / 1e9:.3f} GFLOPs/token cible")
    # SEQ_LEN - 1: le dernier token de chaque chunk n'a pas de target depuis le fix
    # du double shift (voir BinDataset.__getitem__ et NaylisLlamaForCausalLM.forward).
    tokens_per_step = args.batch_size * args.grad_accum * (SEQ_LEN - 1)

    mfu_cb = MFUCallback(
        tokens_per_step=tokens_per_step,
        flops_per_token=flops_per_token,
        peak_flops=PEAK_FLOPS,
        total_tokens=TOTAL_TOKENS,
        model=model,
    )

    output_dir = f"./ablation_{args.variant}"
    resume_path = try_download_checkpoint(args.variant, args.seed)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=5000,
        save_total_limit=1,
        bf16=True,
        torch_compile=False,
        report_to="none",
        dataloader_num_workers=4,
        seed=args.seed,
        data_seed=args.seed,
    )

    checkpoint_cb = HourlyCheckpointCallback(args.variant, args.seed, interval_sec=CHECKPOINT_INTERVAL_SEC)
    callbacks = [mfu_cb, checkpoint_cb]
    if args.variant == "naylis_graph_moe":
        callbacks.append(NoiseAnnealCallback(model, cutoff_frac=0.15))

    trainer = NaylisTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    trainer.remove_callback(PrinterCallback)

    progress_cb = next(
        (cb for cb in trainer.callback_handler.callbacks if isinstance(cb, ProgressCallback)),
        None
    )
    if progress_cb is not None:
        mfu_cb.progress_bar = progress_cb.training_bar

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(f"./model_{args.variant}")

    # ------------------------------------------------------------
    # Graphiques d'entraînement, resume-aware
    # ------------------------------------------------------------
    # trainer.state.log_history contient déjà tout l'historique (avant ET
    # après reprise) car Trainer recharge le trainer_state.json du
    # checkpoint et continue à l'enrichir. On fusionne quand même avec le
    # trainer_state.json du checkpoint de reprise par sécurité (dédup par
    # step, la version la plus récente l'emporte).
    old_history = load_log_history_from_checkpoint(resume_path)
    merged_history = merge_log_histories(old_history, trainer.state.log_history)

    curves_dir = f"./curves_{args.variant}"
    print(f"\n[Curves] Génération des graphiques ({len(merged_history)} points de log fusionnés)...")
    generate_training_curves(merged_history, curves_dir, run_label=args.variant)
    push_curves_to_hub(curves_dir, args.variant, args.seed)

    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True)
        api.upload_folder(
            folder_path=f"./model_{args.variant}",
            path_in_repo=args.variant,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"Uploadé: {HF_REPO_ID}/{args.variant}")
    except Exception as e:
        print(f"[ATTENTION] Échec upload: {e}")


if __name__ == "__main__":
    main()