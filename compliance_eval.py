"""Núcleo del eval de talent-screening (Campus Recruitment) — AGNÓSTICO al framework MLOps.

Mismo patrón que loan-scoring (`compliance_eval.py`):
  · Carga el dato REAL vía Croissant (§2, mlcroissant — no read_csv a mano) desde el Croissant
    LOCAL `data/campus.croissant.json` (contentUrl = bare `Placement_Data_Full_Class.csv`,
    resuelto relativo al directorio del propio Croissant).
  · Corre la SDK venturalitica: `vl.monitor` abre la sesión + los probes (incl. BOMProbe);
    `vl.enforce` evalúa el OSCAL en dos fases (Art.10 datos / Art.15 modelo).
  · PROMUEVE el bom.json del run a `.venturalitica/bom.json` (ruta que lee el motor Rust).
  · Vuelca `metrics.json` plano `{control_id: value | {value, power}}`.
  NO juzga: el veredicto autoritativo lo pone el motor Rust contra el MISMO OSCAL.

DOBLE HISTORIA HONESTA del escenario:
  1. Gate bloqueante NO CERRABLE: `data-sample-adequacy` (Art.10(5)) = n/500 = 215/500 = 0.43.
     ROJO SIEMPRE — n=215 (76 mujeres) es insuficiente para certificar equidad de selección
     con fiabilidad. Sólo lo cierra adquirir más datos (data_gap), nunca un cambio de código.
  2. Sub-tratamiento que SÍ cierra: `model-feature-leakage`. V1 (`mitigate=False`) incluye
     `salary` (NaN ⇔ Not Placed ⇒ imputa 0 ⇒ fuga total) → leakage=1, ROJO. V2
     (`mitigate=True`) retira `salary` → leakage=0, VERDE. Arco demostrativo rojo→verde.
  3. Equidad: ADVISORY y SUBPODERADA (honesto con n=215; ~23 mujeres en test).

Métricas de datos (Art.10) calculadas EN-EVAL sobre el dataset completo (no del catálogo SDK):
  data-gender-representation = min(M,F)/n   (umbral >= 0.30)
  data-class-balance         = Placed/total (umbral >= 0.40)
  data-sample-adequacy       = n/500        (umbral >= 1.0 — ROJO permanente)
  model-feature-leakage      = 1 iff salary en features del modelo (del indicador de train.py)
Métricas de modelo (Art.15) vía `vl.enforce` sobre la cohort TEST:
  model-accuracy · selection-rate-disparity-gender (advisory) · recall-gap-gender (advisory)
"""

import os

os.environ.setdefault("VENTURALITICA_NO_ANALYTICS", "1")  # sin telemetría en CI

import contextlib
import json
import shutil
import sys
from pathlib import Path

import mlcroissant as mlc
import pandas as pd
import yaml

import venturalitica as vl

CROISSANT = "data/campus.croissant.json"
RECORD_SET = "candidates"        # @id del RecordSet en el Croissant local
OSCAL = "shared_data/policies/assessment_plan.oscal.yaml"
PARAMS = "params.yaml"
METRICS = "metrics.json"
BOM_ROOT = ".venturalitica/bom.json"
RUNS_DIR = Path(".venturalitica/runs")

# Convenio de columnas internas.
TARGET = "status"        # 1 = Placed, 0 = Not Placed
PREDICTION = "prediction"
GENDER = "gender"        # 1 = M, 0 = F

# Umbral mínimo de adecuación muestral: n < 500 señala riesgo Art.10(5).
_MIN_SAMPLE_ADEQUATE = 500

# Política OSCAL embebida — fallback para ejecutar el eval de forma autocontenida (sin
# `venth compile`). En la ejecución normal del escenario, `venth compile` genera el OSCAL en
# shared_data/policies/ y ESE es el que usa compliance_eval. Este dict DEBE mantenerse en
# sincronía con venth.yaml manualmente.
_EMBEDDED_POLICY: dict = {
    "component-definition": {
        "metadata": {"title": "Campus Recruitment Compliance Policy (doble historia: fuga + adecuacion)"},
        "components": [
            {
                "control-implementations": [
                    {
                        "implemented-requirements": [
                            {
                                "control-id": "model-accuracy",
                                "description": "Exactitud del clasificador de colocación en el subconjunto test.",
                                "props": [
                                    {"name": "metric_key", "value": "accuracy_score"},
                                    {"name": "threshold", "value": "0.6"},
                                    {"name": "operator", "value": "gt"},
                                    {"name": "severity", "value": "medium"},
                                    {"name": "lifecycle_phase", "value": "validation"},
                                    {"name": "input.target", "value": TARGET},
                                    {"name": "input.prediction", "value": PREDICTION},
                                ],
                            },
                            {
                                "control-id": "selection-rate-disparity-gender",
                                "description": "Paridad demográfica de la decisión por género (advisory).",
                                "props": [
                                    {"name": "metric_key", "value": "demographic_parity_diff"},
                                    {"name": "threshold", "value": "0.10"},
                                    {"name": "operator", "value": "lt"},
                                    {"name": "severity", "value": "medium"},
                                    {"name": "lifecycle_phase", "value": "validation"},
                                    {"name": "input.target", "value": TARGET},
                                    {"name": "input.prediction", "value": PREDICTION},
                                    {"name": "input.dimension", "value": GENDER},
                                ],
                            },
                            {
                                "control-id": "recall-gap-gender",
                                "description": "Gap de recall por género (advisory).",
                                "props": [
                                    {"name": "metric_key", "value": "equal_opportunity_diff"},
                                    {"name": "threshold", "value": "0.15"},
                                    {"name": "operator", "value": "lt"},
                                    {"name": "severity", "value": "medium"},
                                    {"name": "lifecycle_phase", "value": "validation"},
                                    {"name": "input.target", "value": TARGET},
                                    {"name": "input.prediction", "value": PREDICTION},
                                    {"name": "input.dimension", "value": GENDER},
                                ],
                            },
                        ]
                    }
                ]
            }
        ],
    }
}


# -- Carga de datos (Croissant §2) -------------------------------------------------------------


def load_campus(croissant_path: str = CROISSANT) -> pd.DataFrame:
    """Carga Campus Recruitment REAL vía mlcroissant desde el Croissant LOCAL.

    Normaliza al convenio interno: gender 1=M/0=F (int), status 1=Placed/0=Not Placed (int).
    salary queda crudo (NaN para Not Placed); train.py lo imputa a 0.
    """
    ds = mlc.Dataset(jsonld=croissant_path)
    df = pd.DataFrame(list(ds.records(record_set=RECORD_SET)))
    # mlcroissant nombra los campos "<recordset>/<col>" y devuelve texto en bytes.
    df = df.rename(columns=lambda c: c.split("/", 1)[1] if "/" in c else c)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)

    # Codificar la etiqueta: 1 = Placed, 0 = Not Placed.
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Placed").astype(int)
    # Codificar el atributo protegido: 1 = M, 0 = F.
    df[GENDER] = (df[GENDER].astype(str).str.strip() == "M").astype(int)
    # salary numérico (NaN para Not Placed se preserva; train.py imputa 0).
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    n_m = int(df[GENDER].sum())
    n_f = len(df) - n_m
    placed = int(df[TARGET].sum())
    print(
        f"[compliance_eval] n={len(df)}  M={n_m}  F={n_f} (F={n_f / len(df) * 100:.1f}%)  "
        f"Placed={placed} ({placed / len(df) * 100:.1f}%)",
        file=sys.stderr,
    )
    return df


def params() -> dict:
    return yaml.safe_load(open(PARAMS)) or {}


# -- Métricas de datos (Art.10) — sin modelo --------------------------------------------------


def _data_gender_representation(df: pd.DataFrame) -> float:
    """min(M,F)/n — F=35% en Campus Recruitment; umbral orientativo >= 0.30."""
    counts = df[GENDER].value_counts()
    return float(counts.min() / len(df))


def _data_class_balance(df: pd.DataFrame) -> float:
    """Tasa de positivos (Placed/total); Placed=68.8% en Campus Recruitment."""
    return float(df[TARGET].mean())


def _data_sample_adequacy(df: pd.DataFrame) -> float:
    """n/500 — con n=215 da 0.43 → BLOQUEANTE (Art.10(5)). No lo cierra ningún tratamiento."""
    return float(len(df) / _MIN_SAMPLE_ADEQUATE)


# -- Utilidades de métricas -------------------------------------------------------------------


def _metric_entry(result) -> float | dict:
    """Entrada de `metrics.json`: `{value, power}` si el SDK expone el bloque de poder
    (bootstrap, ≥0.6.11), escalar `value` si no. El núcleo Rust acepta ambas formas."""
    value = float(result.actual_value)
    power = getattr(result, "power", None)
    return {"value": value, "power": power} if power else value


def _promote_bom() -> None:
    """Promueve el bom.json que BOMProbe dejó en .venturalitica/runs/<run>/ a la raíz
    .venturalitica/bom.json (la ruta que lee el motor). Elige el run con mtime máximo
    (misma heurística que el CLI push). FAIL-LOUD si no hay ningún bom.json que promover."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob("*/bom.json") if p.parent.name != "latest"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("compliance_eval: no se generó ningún bom.json en .venturalitica/runs/")
    Path(".venturalitica").mkdir(exist_ok=True)
    shutil.copyfile(candidates[-1], BOM_ROOT)
    print(f"bom → {BOM_ROOT} (desde {candidates[-1]})", file=sys.stderr)


def write_metrics(metrics: dict, path: str = METRICS) -> None:
    json.dump(metrics, open(path, "w"), indent=2)


# -- Eval principal ---------------------------------------------------------------------------


def run(build_model, df: pd.DataFrame | None = None, oscal_path: str = OSCAL):
    """Orquesta la evaluación sobre datos REALES (Campus Recruitment):

      1. Carga vía Croissant local si `df` es None.
      2. Lee `seed`/`mitigate` de params.yaml.
      3. Entrena vía `build_model(df, seed, mitigate)` → (cohort_test, model, X_test).
      4. Mide con el venturalitica-sdk: fase training (Art.10) NO-OP (las claves de datos no
         están en el registro del SDK → [Skip]; se inyectan a mano abajo) + fase validation
         (Art.15) sobre la cohort TEST (model-accuracy, paridades advisory).
      5. Añade las métricas de datos manuales (Art.10) + el indicador de fuga.
      6. PROMUEVE el bom.json del run a .venturalitica/bom.json.
      7. Vuelca metrics.json plano.

    Si `oscal_path` no existe (modo autocontenido sin `venth compile`), usa la política embebida.
    Devuelve `(cohort_test, model)`.
    """
    if df is None:
        df = load_campus()
    p = params()
    seed = int(p.get("seed", 42))
    mitigate = bool(p.get("mitigate", False))

    policy = oscal_path if Path(oscal_path).exists() else _EMBEDDED_POLICY

    with contextlib.redirect_stdout(sys.stderr):
        with vl.monitor(name="talent-screening (Campus Recruitment real)", label="venth eval"):
            cohort_test, model, _X_test = build_model(df, seed, mitigate)  # ENTRENAMIENTO (Art.15)

            # Indicador de fuga: lo escribe build_model en el módulo de entrenamiento según
            # `mitigate` (1 iff salary en features). Lo leemos vía sys.modules.
            _mod = sys.modules.get(build_model.__module__, None)
            has_leaky_feature = int(getattr(_mod, "HAS_LEAKY_FEATURE", 0))

            # -- Fase datos (Art.10) sobre el dataset COMPLETO ----------------------------------
            # Intencionadamente un NO-OP: sus claves (sample_adequacy_ratio/leaky_feature_flag…)
            # no están en el registro del SDK → las salta ([Skip]); se inyectan a mano abajo.
            # Se conserva por simetría con la fase modelo.
            data_results = vl.enforce(
                data=df, policy=policy, target=TARGET, gender=GENDER,
                phase="training", strict=False,
            )

            # -- Fase modelo (Art.15) sobre la cohort TEST --------------------------------------
            model_results = vl.enforce(
                data=cohort_test, policy=policy, target=TARGET, prediction=PREDICTION,
                gender=GENDER, phase="validation", strict=False,
            )
        _promote_bom()  # tras cerrar la sesión, el bom.json del run ya existe

    metrics = {r.control_id: _metric_entry(r) for r in (data_results + model_results)}

    # -- Métricas de datos manuales (Art.10) sobre el dataset completo ------------------------
    metrics["data-gender-representation"] = _data_gender_representation(df)
    metrics["data-class-balance"] = _data_class_balance(df)
    metrics["data-sample-adequacy"] = _data_sample_adequacy(df)
    # -- Control de fuga de feature: 1 si salary en features (V1), 0 si no (V2) ---------------
    metrics["model-feature-leakage"] = has_leaky_feature

    n_test = len(cohort_test)
    n_test_f = int((cohort_test[GENDER] == 0).sum()) if GENDER in cohort_test.columns else "?"
    print(
        f"[compliance_eval] datos Art.10: gender-repr={metrics['data-gender-representation']:.3f}  "
        f"class-balance={metrics['data-class-balance']:.3f}  "
        f"sample-adequacy={metrics['data-sample-adequacy']:.3f}  "
        f"(n={len(df)})",
        file=sys.stderr,
    )
    print(
        f"[compliance_eval] cohort test: n={n_test}, F_test={n_test_f}  "
        f"model-feature-leakage={has_leaky_feature}  mitigate={mitigate}",
        file=sys.stderr,
    )

    write_metrics(metrics)
    return cohort_test, model
