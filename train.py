"""Tratamiento de talent-screening — modelo y su VARIANTE (V1/V2). El arco honesto rojo→verde
de la FUGA lo gobierna `mitigate` (reutiliza el parámetro arc de loan-scoring):

  mitigate=False (V1): incluye `salary` como feature. `salary` es NaN exactamente para los
                       candidatos Not Placed → imputar 0 hace que el modelo aprenda la etiqueta
                       de forma trivial (fuga objetivo total) → exactitud inflada (~1.0) y
                       model-feature-leakage = 1 (ROJO, restricción "< 1" falla).
  mitigate=True  (V2): RETIRA `salary` (el tratamiento ISO 23894 §6.5, versionado en git). El
                       modelo aprende de variables legítimas (notas académicas, experiencia) →
                       exactitud honesta (~0.80-0.87) y model-feature-leakage = 0 (VERDE).

El control bloqueante data-sample-adequacy (n/500 = 0.43) sigue ROJO en AMBAS versiones: la
muestra no se cierra con ningún cambio de código (sólo adquirir más datos → data_gap).

build_model(df, seed, mitigate) -> (cohort_test, modelo, X_test). La cohort devuelta es el
subconjunto TEST (70/30 split determinista con seed fijo) — evaluación honesta en datos no
vistos, mismo contrato que los evals loan/retinopathy. El indicador de fuga vive en el atributo
de módulo HAS_LEAKY_FEATURE, que build_model fija según `mitigate` (1 iff salary en features);
compliance_eval lo lee a través de sys.modules para el control model-feature-leakage."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "status"       # 1 = Placed, 0 = Not Placed (codificado en compliance_eval)
GENDER = "gender"       # 1 = M, 0 = F (codificado en compliance_eval)

# Columnas siempre excluidas del entrenamiento: etiqueta, predicción, protegido e índice.
_DROP_ALWAYS = [TARGET, "prediction", GENDER, "sl_no"]
# La feature fugada: presente en V1 (mitigate=False), retirada en V2 (mitigate=True).
_LEAKY_FEATURE = "salary"

# Indicador de fuga para el control model-feature-leakage (1 = hay fuga, 0 = no). build_model lo
# fija según `mitigate`; compliance_eval lo lee del módulo. Default 1 (V1) por simetría con el
# test-resource (HAS_LEAKY_FEATURE), pero el valor autoritativo lo escribe build_model.
HAS_LEAKY_FEATURE: int = 1


def build_model(df: pd.DataFrame, seed: int, mitigate: bool = False):
    """Entrena el cribador de colocación (GradientBoosting). El tratamiento es `mitigate`:

      · mitigate=False (V1): incluye `salary` (imputando NaN→0). Fuga objetivo total →
        exactitud inflada (~1.0). HAS_LEAKY_FEATURE = 1.
      · mitigate=True  (V2): retira `salary`. Exactitud honesta (~0.80-0.87).
        HAS_LEAKY_FEATURE = 0.

    Devuelve `(cohort TEST con prediction, modelo sklearn, X_test)`. Split determinista
    70/30 con seed fijo (idéntico entre V1/V2 para comparabilidad).
    """
    global HAS_LEAKY_FEATURE
    HAS_LEAKY_FEATURE = 0 if mitigate else 1

    # Imputar la feature fugada NaN→0 (patrón de fuga aún más obvio: salary=0 ⇔ Not Placed).
    df_work = df.copy()
    df_work[_LEAKY_FEATURE] = df_work[_LEAKY_FEATURE].fillna(0.0).astype(float)

    drop_cols = list(_DROP_ALWAYS)
    if mitigate:
        drop_cols.append(_LEAKY_FEATURE)   # el TRATAMIENTO: retirar la feature fugada
    drop_cols = [c for c in drop_cols if c in df_work.columns]

    X = pd.get_dummies(df_work.drop(columns=drop_cols)).astype(float)
    y = df_work[TARGET].astype(int)

    # Split determinista 70/30 (mismo seed entre V1/V2).
    Xtr, Xte, ytr, _yte = train_test_split(X, y, test_size=0.30, random_state=seed)

    # GradientBoosting: con salary aprende la fuga casi perfectamente (V1); sin salary aprende
    # de méritos legítimos (V2). max_depth=2 evita sobreajuste severo con ~150 filas de train.
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed),
    ).fit(Xtr, ytr)

    # Cohort de evaluación = subconjunto TEST (datos no vistos).
    cohort = df.loc[Xte.index].copy()
    cohort[_LEAKY_FEATURE] = df_work.loc[Xte.index, _LEAKY_FEATURE]  # salary imputada
    cohort["prediction"] = model.predict(Xte)
    return cohort, model, Xte
