"""Stage `evaluate` (DVC): corre el eval agnóstico sobre Campus Recruitment (cargado vía
Croissant) y persiste el modelo como out cacheado (Art.15). La medición (SDK + métricas custom
+ BOM) vive en compliance_eval; el tratamiento (variante V1/V2 vía `mitigate`) en train.py."""

import joblib

import compliance_eval
import train

df = compliance_eval.load_campus()
_, model = compliance_eval.run(train.build_model, df)
joblib.dump(model, "model.pkl")
