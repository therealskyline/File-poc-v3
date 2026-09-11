"""
Naylis Ablation - Benchmark lm-eval

Benchmark des modèles Naylis téléchargés depuis HuggingFace.
Tâches: hellaswag, arc_easy, arc_challenge, piqa, boolq, copa,
        winogrande, sciq, openbookqa

Ce script s'appuie directement sur naylis_ablation2.py (le script
d'entraînement "Graph MoE Edition") pour reconstruire les modèles à
l'identique:
  - Le tokenizer n'est PAS sauvegardé avec les checkpoints (le Trainer
    n'en a jamais reçu), donc on utilise TOKENIZER_NAME importé depuis
    naylis_ablation2.py ("HuggingFaceTB/cosmo2-tokenizer").
  - "vanilla" et "vanilla_thin_ffn" sont tous les deux des LlamaForCausalLM
    standards (pas de graph memory) qui ne diffèrent que par
    intermediate_size — cette valeur est déjà dans le config.json
    sauvegardé, donc pas besoin de la recalculer ici.
  - NaylisLlamaForCausalLM.__init__ exige un argument positionnel
    `variant` qui n'existe pas dans config.json, et ce nom collisionne
    avec le kwarg réservé `variant` de transformers.from_pretrained
    (qui sert à choisir un fichier de poids alternatif). On ne peut donc
    pas passer par .from_pretrained(..., variant=variant) ni par
    AutoModelForCausalLM.register(...) tel quel: on instancie la bonne
    classe nous-mêmes (LlamaForCausalLM pour "vanilla"/"vanilla_thin_ffn",
    NaylisLlamaForCausalLM sinon) puis on charge les poids à la main.
    lm-eval accepte ensuite un nn.Module déjà instancié via HFLM(pretrained=...).

naylis_ablation2.py doit être présent dans le même répertoire que ce
script.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import torch
import numpy as np

# ============================================================
# IMPORT DES CLASSES / CONSTANTES DEPUIS LE SCRIPT D'ENTRAÎNEMENT
# ============================================================
from naylis_ablation2 import (
    NaylisLlamaForCausalLM,
    TOKENIZER_NAME,
    MEM_SIZE,
    GRAPH_RATIO,
)
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# ============================================================
# CONFIGURATION
# ============================================================
# Le token de naylis_ablation2.py a fuité dans cette conversation et doit
# être considéré compromis: révoque-le sur huggingface.co/settings/tokens,
# régénères-en un, puis passe-le ici via variable d'environnement.
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = "TheRealSkyline/naylis_ablation_300M"
HF_REPO_TYPE = "dataset"
SEED = 257

# Variants supportés par naylis_ablation2.py. Seuls vanilla,
# vanilla_thin_ffn et naylis_graph_moe ont effectivement été entraînés
# pour l'instant (voir --models plus bas), mais on garde la liste
# complète pour --models et la validation d'argparse.
VARIANTS = [
    "vanilla",
    "vanilla_thin_ffn",
    "naylis_global",
    "naylis_layeroff",
    "naylis_graph_moe",
    "naylis_local",
]

# Variants réellement disponibles sur le Hub à ce jour.
TRAINED_VARIANTS = [
    "vanilla",
    "vanilla_thin_ffn",
    "naylis_graph_moe",
]

# Variants qui sont un LlamaForCausalLM standard (pas de graph memory).
# Seul intermediate_size change entre les deux, et c'est déjà encodé
# dans le config.json sauvegardé par le Trainer.
VANILLA_VARIANTS = ("vanilla", "vanilla_thin_ffn")

TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "boolq",
    "copa",
    "winogrande",
    "sciq",
    "openbookqa",
]

_TOKENIZER_CACHE = None


def get_tokenizer():
    """Charge (une seule fois) le tokenizer fixe utilisé à l'entraînement."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        print(f"[Tokenizer] Chargement de {TOKENIZER_NAME}...")
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER_CACHE


# ============================================================
# TÉLÉCHARGEMENT DES MODÈLES DEPUIS HUGGINGFACE
# ============================================================
def download_model_from_hf(variant, seed=SEED):
    """
    Télécharge le checkpoint final d'un variant depuis le repo HF.
    Les poids sont uploadés à path_in_repo=variant (voir
    naylis_ablation2.py: api.upload_folder(path_in_repo=args.variant, ...)).
    Fallback: checkpoint intermédiaire à {variant}_seed{seed}/checkpoint.
    """
    from huggingface_hub import snapshot_download

    print(f"\n[Download] Téléchargement du modèle {variant} depuis {HF_REPO_ID}...")

    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN or None,
            allow_patterns=f"{variant}/*",
        )
        model_path = os.path.join(local_dir, variant)

        if os.path.isdir(model_path) and os.listdir(model_path):
            print(f"[Download] ✓ Modèle téléchargé: {model_path}")
            print(f"[Download]   Fichiers: {os.listdir(model_path)}")
            return model_path
        else:
            print(f"[Download] ✗ Dossier vide pour {variant}")
    except Exception as e:
        print(f"[Download] ✗ Erreur: {e}")

    # Fallback: checkpoint intermédiaire du Trainer
    print("[Download] Essai de fallback sur le checkpoint intermédiaire...")
    try:
        key = f"{variant}_seed{seed}"
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN or None,
            allow_patterns=f"{key}/checkpoint/*",
        )
        model_path = os.path.join(local_dir, key, "checkpoint")
        if os.path.isdir(model_path) and os.listdir(model_path):
            print(f"[Download] ✓ Checkpoint trouvé: {model_path}")
            print(f"[Download]   Fichiers: {os.listdir(model_path)}")
            return model_path
    except Exception as e2:
        print(f"[Download] ✗ Fallback aussi échoué: {e2}")

    return None


# ============================================================
# CONSTRUCTION DU MODÈLE (même logique que naylis_ablation2.py::main)
# ============================================================
def _find_weights_file(model_path):
    """Repère le fichier de poids dans le dossier téléchargé."""
    candidates = ["model.safetensors", "pytorch_model.bin"]
    for name in candidates:
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            return p
    # Cas shardé (peu probable pour un 300M, mais on couvre le cas)
    index_candidates = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
    for name in index_candidates:
        if os.path.isfile(os.path.join(model_path, name)):
            return os.path.join(model_path, name)
    raise FileNotFoundError(f"Aucun fichier de poids trouvé dans {model_path}")


def _load_state_dict(weights_path):
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    if weights_path.endswith(".index.json"):
        # Poids shardés: on charge chaque shard référencé par l'index.
        import glob
        model_dir = os.path.dirname(weights_path)
        state_dict = {}
        if weights_path.endswith(".safetensors.index.json"):
            from safetensors.torch import load_file
            for shard in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
                state_dict.update(load_file(shard))
        else:
            for shard in sorted(glob.glob(os.path.join(model_dir, "pytorch_model-*.bin"))):
                state_dict.update(torch.load(shard, map_location="cpu"))
        return state_dict
    # .bin classique
    return torch.load(weights_path, map_location="cpu")


def load_model(model_path, variant, device="cuda"):
    """
    Reconstruit le bon modèle à partir d'un checkpoint, en respectant
    exactement la logique de naylis_ablation2.py:
      - "vanilla" et "vanilla_thin_ffn" -> LlamaForCausalLM nu
        (intermediate_size différent, mais déjà encodé dans config.json)
      - tous les autres variants -> NaylisLlamaForCausalLM(variant=...)

    IMPORTANT: on n'utilise PAS `.from_pretrained(..., variant=variant)`.
    `variant` est un kwarg RÉSERVÉ par transformers.PreTrainedModel.from_pretrained
    (il sert à choisir un fichier de poids alternatif, ex: model.fp16.safetensors).
    Le passer collisionne avec notre propre `variant` custom (naylis_graph_moe,
    etc.) et fait chercher à transformers un fichier
    "model.<notre_variant>.safetensors" qui n'existe pas.
    On construit donc le modèle nous-mêmes, puis on charge les poids à la main.
    """
    config = LlamaConfig.from_pretrained(model_path)

    if variant in VANILLA_VARIANTS:
        model = LlamaForCausalLM(config)
    else:
        model = NaylisLlamaForCausalLM(
            config, variant=variant, mem_size=MEM_SIZE, graph_ratio=GRAPH_RATIO,
        )

    weights_path = _find_weights_file(model_path)
    state_dict = _load_state_dict(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] clés manquantes lors du chargement: {missing}")
    if unexpected:
        print(f"[Warning] clés inattendues lors du chargement: {unexpected}")

    model = model.to(torch.float16)
    model.eval()
    model = model.to(device if torch.cuda.is_available() else "cpu")
    return model


# ============================================================
# BENCHMARK
# ============================================================
def benchmark_model(model_path, variant, tasks, num_fewshot=0, batch_size=8, device="cuda"):
    """Benchmark un modèle avec lm-eval."""
    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator
    except ImportError:
        print("\n❌ lm-eval n'est pas installé.")
        print("   pip install lm-eval")
        return None

    print(f"\n{'='*70}")
    print(f"Benchmark: {variant}")
    print(f"Chemin: {model_path}")
    print(f"Tâches: {', '.join(tasks)}")
    print(f"{'='*70}")

    try:
        model = load_model(model_path, variant, device=device)
        tokenizer = get_tokenizer()
    except Exception as e:
        print(f"[Error] Impossible de charger le modèle: {e}")
        return None

    try:
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device if torch.cuda.is_available() else "cpu",
        )
    except Exception as e:
        print(f"[Error] Impossible d'initialiser HFLM: {e}")
        return None

    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"[Error] Échec du benchmark: {e}")
        return None

    # Extraction des résultats
    results_dict = {}
    for task_name in tasks:
        if task_name in results["results"]:
            task_results = results["results"][task_name]

            if "acc_norm,none" in task_results:
                acc = task_results["acc_norm,none"]
                metric_name = "acc_norm"
            elif "acc,none" in task_results:
                acc = task_results["acc,none"]
                metric_name = "acc"
            else:
                acc = None
                metric_name = "N/A"

            results_dict[task_name] = {"accuracy": acc, "metric": metric_name}
            print(f"  {task_name:<20} {acc:.4f} ({metric_name})" if acc is not None
                  else f"  {task_name:<20} N/A")
        else:
            results_dict[task_name] = {"accuracy": None, "metric": "N/A"}
            print(f"  {task_name:<20} non évalué")

    return results_dict


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================
def print_results_table(all_results, tasks):
    model_names = list(all_results.keys())
    col_width = 18

    print(f"\n{'='*80}")
    print("RÉSULTATS BENCHMARK LM-EVAL")
    print(f"{'='*80}")

    header = f"{'Tâche':<20}"
    for name in model_names:
        header += f"{name:<{col_width}}"
    print(header)
    print("-" * 80)

    for task in tasks:
        row = f"{task:<20}"
        for name in model_names:
            acc = all_results.get(name, {}).get(task, {}).get("accuracy")
            row += f"{acc:.4f}{'':<{col_width-6}}" if acc is not None else f"{'N/A':<{col_width}}"
        print(row)

    print("-" * 80)

    row = f"{'MOYENNE':<20}"
    for name in model_names:
        accs = [r["accuracy"] for r in all_results.get(name, {}).values() if r["accuracy"] is not None]
        if accs:
            row += f"{np.mean(accs):.4f}{'':<{col_width-6}}"
        else:
            row += f"{'N/A':<{col_width}}"
    print(row)
    print(f"{'='*80}")


def save_results(all_results, output_path="benchmark_results.json"):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Résultats sauvegardés dans {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark lm-eval des modèles Naylis")
    parser.add_argument(
        "--models", nargs="+", default=TRAINED_VARIANTS,
        choices=VARIANTS, help="Variants à benchmarker",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Tâches à évaluer")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Utiliser les modèles locaux (./model_<variant>) au lieu de télécharger depuis HF",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NAYLIS ABLATION - BENCHMARK LM-EVAL")
    print("=" * 70)
    print(f"Modèles: {args.models}")
    print(f"Tâches: {args.tasks}")
    print(f"Few-shot: {args.num_fewshot}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tokenizer: {TOKENIZER_NAME}")

    try:
        import lm_eval
        print(f"\n✓ lm-eval version: {lm_eval.__version__}")
    except ImportError:
        print("\n❌ lm-eval n'est pas installé. pip install lm-eval")
        return

    all_results = {}

    for variant in args.models:
        print(f"\n{'#'*70}")
        print(f"# TRAITER: {variant}")
        print(f"{'#'*70}")

        if args.skip_download:
            model_path = f"./model_{variant}"
            if not os.path.exists(model_path):
                print(f"⚠️  Modèle local non trouvé: {model_path}")
                continue
        else:
            model_path = download_model_from_hf(variant, seed=SEED)
            if model_path is None:
                print(f"⚠️  Impossible de récupérer le modèle {variant}")
                continue

        results = benchmark_model(
            model_path=model_path,
            variant=variant,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
        )

        if results is not None:
            all_results[variant] = results

    if all_results:
        print_results_table(all_results, args.tasks)
        save_results(all_results, args.output)
    else:
        print("\n⚠️  Aucun résultat à afficher.")


if __name__ == "__main__":
    main()
