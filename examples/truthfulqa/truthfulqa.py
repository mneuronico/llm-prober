import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict

def read_truthfulqa(
    dir_path: str = ".",
    csv_path: str | None = None,
    n: int | None = None,
    stratify: str | None = None,   # None | "type" | "category" | "both"
    priority: str = "type",        # "type" o "category" (solo aplica si stratify == "both")
    seed: int | None = None,
    id_prefix: str = "truthfulqa", # ← nuevo: prefijo para IDs
):
    """
    Lee TruthfulQA.csv y devuelve **una lista** de items:
        [
          {
            "id": "<id_prefix>_0000",
            "type": "adversarial" | "non_adversarial",
            "prompt": <Question>,
            "correct_answers": [ ... ],
            "incorrect_answers": [ ... ],
          },
          ...
        ]
    Soporta muestreo estratificado por type/category/both y seed reproducible.
    """
    # --- Local helpers ---
    def _split_list_field(s: pd.Series | str | None):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return []
        if isinstance(s, pd.Series):
            s = s.astype(str).fillna("")
        if isinstance(s, str):
            raw = [x.strip() for x in s.split(";")]
        else:
            raw = []
        seen = set(); out = []
        for x in raw:
            x_clean = x.strip(" \"'").strip()
            if not x_clean:
                continue
            key = x_clean.lower()
            if key not in seen:
                seen.add(key)
                out.append(x_clean)
        return out

    def _to_bool_adversarial(type_val: str) -> bool:
        t = (type_val or "").strip().lower()
        t_norm = re.sub(r"[\s_]+", " ", t).strip()
        if re.fullmatch(r"(non[-\s]?adversarial)", t_norm):
            return False
        if re.fullmatch(r"(adversarial)", t_norm):
            return True
        if "non" in t_norm and "adversarial" in t_norm:
            return False
        if "adversarial" in t_norm:
            return True
        return False
    
    # --- Carga ---
    path = csv_path if csv_path is not None else os.path.join(dir_path, "TruthfulQA.csv")
    df = pd.read_csv(path, dtype=str).fillna("")

    expected_cols = {
        "Type", "Category", "Question",
        "Best Answer", "Best Incorrect Answer",
        "Correct Answers", "Incorrect Answers", "Source"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Normalización: bucket adversarial / non_adversarial
    df["_is_adv"] = df["Type"].apply(_to_bool_adversarial)
    df["_bucket"] = np.where(df["_is_adv"], "adversarial", "non_adversarial")

    # --- Muestreo (igual que antes) ---
    rng = np.random.default_rng(seed)
    total = len(df)
    if n is None or n <= 0 or n >= total:
        df_sampled = df.copy()
    else:
        if stratify is None:
            df_sampled = df.sample(n=n, random_state=seed)
        else:
            if stratify not in {"type", "category", "both"}:
                raise ValueError("stratify debe ser None, 'type', 'category' o 'both'.")

            if stratify == "type":
                strat_key = df["_bucket"]
            elif stratify == "category":
                strat_key = df["Category"].astype(str)
            else:  # "both"
                if priority not in {"type", "category"}:
                    raise ValueError("priority debe ser 'type' o 'category' cuando stratify == 'both'.")
                if priority == "type":
                    strat_key = list(zip(df["_bucket"], df["Category"].astype(str)))
                else:
                    strat_key = list(zip(df["Category"].astype(str), df["_bucket"]))
                strat_key = pd.Series(strat_key)

            df_tmp = df.copy()
            df_tmp["_strat"] = strat_key
            counts = df_tmp["_strat"].value_counts().sort_index()
            proportions = counts / counts.sum()
            target_counts = (proportions * n).to_dict()

            assigned = {k: int(np.floor(v)) for k, v in target_counts.items()}
            remainder = n - sum(assigned.values())
            if remainder > 0:
                fracs = {k: target_counts[k] - assigned[k] for k in target_counts}
                for k in sorted(fracs, key=lambda x: fracs[x], reverse=True)[:remainder]:
                    assigned[k] += 1

            parts = []
            for k, need in assigned.items():
                if need <= 0:
                    continue
                grp = df_tmp[df_tmp["_strat"] == k]
                take = min(need, len(grp))
                if take > 0:
                    idx = rng.choice(grp.index.values, size=take, replace=False)
                    parts.append(df.loc[idx])
            df_sampled = pd.concat(parts, axis=0) if parts else df.head(0)

    if len(df_sampled) > 0:
        df_sampled = df_sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Construcción del output: lista + IDs con prefijo configurable ---
    out_list = []
    global_counter = 0  # ← contador global para IDs

    for _, row in df_sampled.iterrows():
        bucket = row["_bucket"]  # "adversarial" o "non_adversarial"
        prompt = row["Question"].strip()

        correct = []
        if row["Best Answer"].strip():
            correct.append(row["Best Answer"].strip())
        correct += _split_list_field(row["Correct Answers"])

        incorrect = []
        if row["Best Incorrect Answer"].strip():
            incorrect.append(row["Best Incorrect Answer"].strip())
        incorrect += _split_list_field(row["Incorrect Answers"])

        # dedup preservando orden
        def dedup(seq):
            seen = set(); out = []
            for x in seq:
                x_norm = x.strip()
                if not x_norm:
                    continue
                key = x_norm.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(x_norm)
            return out

        correct = dedup(correct)
        incorrect = dedup(incorrect)

        ex_id = f"{id_prefix}_{global_counter:04d}"
        global_counter += 1

        out_list.append({
            "id": ex_id,            # ← ahora usa el prefijo configurable
            "type": bucket,
            "prompt": prompt,
            "correct_answers": correct,
            "incorrect_answers": incorrect,
        })

    return out_list
