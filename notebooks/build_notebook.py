"""Genera notebooks/01_experimentacion.ipynb (el notebook se ejecuta después con nbconvert).

Uso: python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Reconocimiento de objetos y ángulo con ML clásico (COIL-100)

Notebook reproducible de punta a punta: datos → análisis exploratorio → protocolos A-D → comparación con baseline y
referencia moderna → análisis de errores → robustez → exportación del artefacto. La lógica vive en `src/vision/`;
el notebook la llama y muestra los resultados. El barrido completo está registrado en MLflow (`mlruns/mlflow.db`) y
resumido en `eval/vision_summary.csv`. Con `RERUN_SWEEP = False` el notebook lee esos resultados en lugar de repetir
el barrido (unos 60 minutos en CPU).
""")
code("""
import os, sys, json, hashlib
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
RERUN_SWEEP = False
SEED = 42
""")
md("## 1. Datos: descarga, verificación y versión")
code("""
import subprocess
zip_path = Path("../datos_coil100/coil-100.zip")
args = [sys.executable, "data/download.py", "--out", "data/raw"] + (["--zip", str(zip_path)] if zip_path.exists() else [])
print(subprocess.run(args, capture_output=True, text=True).stdout)
from src.vision.data import load_index, load_images
from src.vision.sweep import DATA_SHA, load_all
index, images, feats = load_all()
print("SHA-256 del ZIP oficial:", DATA_SHA)
print(index.shape, "objetos:", index.obj_id.nunique(), "vistas por objeto:", index.groupby("obj_id").size().unique())
""")
md("""
## 2. Análisis exploratorio

COIL-100 tiene 7.200 imágenes RGB de 128×128: 100 objetos y 72 vistas por objeto, una cada 5°. El objeto gira sobre una
plataforma (eje vertical): la rotación es **fuera del plano** de la imagen, así que el ángulo no se puede obtener girando
la foto; hay que aprenderlo por objeto. El fondo es negro, la luz controlada y la cámara está a una sola elevación.
""")
code("""
fig, axes = plt.subplots(4, 9, figsize=(13, 6))
for r, obj in enumerate([5, 47, 67, 30]):
    for c, ang in enumerate(range(0, 360, 40)):
        axes[r, c].imshow(images[(index.obj_id == obj) & (index.angle == ang)][0]); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(f"{ang}°", fontsize=8)
    axes[r, 0].text(-40, 64, f"obj {obj}", fontsize=9)
plt.suptitle("Vistas cada 40°: asimétrico (5), simétrico total (47), simétrico 180° (67), asimétrico (30)"); plt.tight_layout(); plt.show()
""")
code("""
fg = (images.max(axis=3) > 20).mean(axis=(1, 2))
print("Fracción de píxeles de objeto: media %.2f, p1 %.2f, p99 %.2f" % (fg.mean(), np.percentile(fg, 1), np.percentile(fg, 99)))
plt.figure(figsize=(5, 3)); plt.hist(fg, bins=40); plt.xlabel("fracción de objeto en la imagen"); plt.ylabel("imágenes"); plt.show()
""")
md("""
### Simetría

Para cada objeto se mide cuánto cambia (distancia en el espacio de ángulo) al girarlo 90° y 180°, comparado con el
objeto mediano del catálogo. Simétrico total: girarlo 90° le cambia menos del 30 % de lo que cambia el objeto mediano.
Simétrico de 180°: girarlo 180° le cambia menos del 40 % de lo que le cambia girarlo 90°.
""")
code("""
from src.vision.model import ModelConfig, VisionModel
full = VisionModel(ModelConfig(classifier="knn", calibrate=False)).fit(None, index.obj_id, index.angle, index.path, precomputed=feats)
sym = pd.Series(full.symmetry)
print(sym.value_counts().rename({360: "asimétrico", 180: "simétrico 180°", 0: "simétrico total"}))
print("simétricos totales:", sorted(sym[sym == 0].index.tolist())); print("simétricos 180°:", sorted(sym[sym == 180].index.tolist()))
""")
md("""
### Qué condiciona el diseño y en qué difiere de una planta

| Propiedad de COIL-100 | Consecuencia en el diseño | Planta real |
|---|---|---|
| Rotación fuera del plano | Ángulo aprendido por objeto (vecinos entre sus vistas) | Igual si el objeto gira sobre la cinta |
| Ángulo circular | Error con distancia circular; media y dispersión circulares | Igual |
| Objetos simétricos | Modo de ángulo `exact` / `mod180` / `any` y métrica con simetría | Igual: la pinza no distingue giros equivalentes |
| Fondo negro, luz fija, objeto centrado | Control de calidad por desenfoque, exposición y fracción de objeto | Fondo y luz variables, varios objetos: haría falta segmentación |
| Una elevación de cámara | Sin variación de inclinación | Objetos volcados o apoyados distinto |
| Vistas cada 5° | Con datos cada 10° (A) el vecino correcto queda a ≤ 5° | Alta de productos: pedir vistas cada 10-15° |
""")
md("""
## 3. Protocolos y validación sin tocar la prueba

| Protocolo | Entrenamiento | Prueba | Validación interna (solo vistas de entrenamiento) |
|---|---|---|---|
| A | 0°, 10°, 20°… | 5°, 15°, 25°… | entrenar en múltiplos de 20°, validar en 10°, 30°… |
| B | todo menos un arco de 60° por objeto | el arco | otro arco contiguo de 60° dentro de lo visto |
| C | una vista cada 30° | las 60 restantes | dejar una vista fuera (12 folds) |
| D | 90 objetos (vistas cada 10°) | vistas a 5° de los 90 + 10 objetos completos | 10 de los conocidos como desconocidos simulados |

Los umbrales de la política (puerta de desconocidos y puntuación de agarre) se eligen **solo** en validación con un
riesgo de agarre erróneo ≤ 1 %; la prueba se evalúa una vez. Agarre correcto = clase correcta y error angular
(con simetría) ≤ 10°, la tolerancia del robot.
""")
code("""
from src.vision.protocols import make_split, save_splits
splits = {p: make_split(index, p) for p in "ABCD"}
for p, s in splits.items(): print(p, "train", len(s.train_idx), "test", len(s.test_idx), {k: (v if k != "arc_start" else "...") for k, v in s.meta.items()})
""")
md("## 4. Barrido: 4 clasificadores × 3 combinaciones de características, baseline y ResNet-50")
code("""
if RERUN_SWEEP:
    Path("eval/vision_summary.csv").unlink(missing_ok=True)
    subprocess.run([sys.executable, "-m", "src.vision.sweep", "--stage", "grid"], check=True)
    subprocess.run([sys.executable, "-m", "src.vision.sweep", "--stage", "angle"], check=True)
summary = pd.read_csv("eval/vision_summary.csv")
cols = ["protocol", "family", "name", "val_coverage", "val_wrong_pick_rate", "test_class_acc", "test_angle_within_tol_sym",
        "test_angle_median_err_sym", "test_ece", "test_coverage", "test_wrong_pick_rate", "latency_ms"]
grid = summary[summary.family != "angle"][cols]
grid.round(4)
""")
code("""
mean_score = summary[summary.family == "classical"].groupby("name")["selection_score"].mean().sort_values(ascending=False)
print("Selección en validación (media de A-D de cobertura − 10 × agarre erróneo):"); print(mean_score.round(4).head(6))
best_name = mean_score.index[0]; print("Mejor clásico:", best_name)
""")
md("### Comparación obligatoria: baseline, mejor clásico y referencia moderna")
code("""
ref = summary[summary.family == "reference"].groupby("protocol").apply(lambda g: g.sort_values("test_class_acc").iloc[-1]).reset_index(drop=True)
rows = []
for p in "ABCD":
    for fam, name in [("baseline", "1nn-pixels"), ("classical", best_name)]:
        r = summary[(summary.protocol == p) & (summary.family == fam) & (summary.name == name)].iloc[0]; rows.append(r)
    rows.append(ref[ref.protocol == p].iloc[0])
cmp = pd.DataFrame(rows)[["protocol", "family", "name", "test_class_acc", "test_angle_within_tol_sym", "test_coverage", "test_wrong_pick_rate", "latency_ms"]]
cmp.round(4)
""")
code("""
import time, joblib, io
from src.vision.reference_cnn import embed
t0 = time.perf_counter(); _ = embed(images[:64]); cnn_ms = (time.perf_counter() - t0) * 1000 / 64
print(f"ResNet-50 en CPU: {cnn_ms:.1f} ms por imagen (más la extracción de características clásicas, que no hace falta)")
buf = io.BytesIO(); joblib.dump(full, buf); print(f"Tamaño de un modelo clásico serializado: {buf.tell()/1e6:.1f} MB; ResNet-50: ~100 MB de pesos")
""")
md("## 5. Ángulo: vecinos frente a kernel ridge")
code("""
ang = summary[summary.family == "angle"]
ang.pivot_table(index="name", columns="protocol", values=["test_angle_within_tol_sym", "val_coverage"]).round(3) if len(ang) else "sin etapa de ángulo"
""")
md("## 6. Confianza: calibración, riesgo-cobertura y rechazo (mejor clásico)")
code("""
from src.vision.experiments import run_protocol, accepted_mask, pick_score
from src.vision.metrics import risk_coverage, ece, auroc
from src.vision.sweep import best_classical
cfg = best_classical()
res = {p: run_protocol(p, cfg, index, feats, DATA_SHA, mlflow_uri=None) for p in "AD"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p, r in res.items():
    t = r["test"]; known = t[t.known]
    bins = np.linspace(0, 1, 11); idx = np.digitize(known.prob, bins) - 1
    xs = [known.prob[idx == b].mean() for b in range(10) if (idx == b).any()]; ys = [known.cls_ok[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[0].plot(xs, ys, "o-", label=f"{p} (ECE {ece(known.prob, known.cls_ok):.3f})")
    good = (t.cls_ok & (t.err_sym <= 10)).to_numpy(); cov, risk = risk_coverage(pick_score(t, r["thresholds"]), good)
    ax[1].plot(cov, risk, label=p)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("confianza de clase"); ax[0].set_ylabel("acierto"); ax[0].set_title("Diagrama de fiabilidad"); ax[0].legend()
ax[1].axhline(0.01, color="k", ls="--", lw=0.8); ax[1].set_xlabel("cobertura (agarres automáticos)"); ax[1].set_ylabel("riesgo (agarres erróneos)"); ax[1].set_ylim(0, 0.2); ax[1].set_title("Riesgo-cobertura en prueba"); ax[1].legend()
d = res["D"]["test"]; ax[2].hist(d[d.known].ood, bins=40, alpha=.6, label="conocidos"); ax[2].hist(d[~d.known].ood, bins=40, alpha=.6, label="desconocidos")
ax[2].axvline(res["D"]["thresholds"]["max_ood"], color="k", ls="--"); ax[2].set_title(f"Distancia al catálogo (D), AUROC {auroc(-d[d.known].ood, -d[~d.known].ood):.3f}"); ax[2].legend()
plt.tight_layout(); plt.savefig("docs/img/confianza.png", dpi=110); plt.show()
""")
md("""
La confianza es una base válida para decidir si (1) está calibrada (ECE bajo), (2) al exigir más confianza el error de
lo aceptado baja (curva riesgo-cobertura decreciente) y (3) separa conocidos de desconocidos (AUROC alto). La política
usa la puntuación de agarre con el umbral elegido en validación.
""")
md("## 7. Análisis de errores")
code("""
t = res["A"]["test"]; known = t[t.known]
conf = known[~known.cls_ok].groupby(["true_obj", "pred_obj"]).size().sort_values(ascending=False).head(8)
print("Pares de clase más confundidos (A):"); print(conf)
err_obj = known[known.cls_ok].groupby("true_obj").err_sym.median().sort_values(ascending=False).head(8)
print("Objetos con mayor error angular mediano (A, con simetría):"); print(err_obj.round(1))
b = run_protocol("B", cfg, index, feats, DATA_SHA, mlflow_uri=None)["test"]
b["pos"] = [(a - s) % 360 for a, s in zip(b.true_angle, [make_split(index, "B").meta["arc_start"][o] for o in b.true_obj])]
plt.figure(figsize=(6, 3)); b.groupby("pos").err_sym.median().plot(marker="o"); plt.xlabel("posición dentro del arco ciego (°)"); plt.ylabel("error mediano (°)"); plt.title("B: el error crece hacia el centro del arco"); plt.show()
""")
md("## 8. Robustez: imágenes degradadas (conjunto de prueba A)")
code("""
import cv2
from src.vision.features import build_features
m = res["A"]["model"]; te = make_split(index, "A").test_idx[::6]
def degrade(imgs, kind):
    if kind == "brillo -40%": return np.clip(imgs * 0.6, 0, 255).astype(np.uint8)
    if kind == "ruido σ=15": return np.clip(imgs + np.random.default_rng(0).normal(0, 15, imgs.shape), 0, 255).astype(np.uint8)
    if kind == "desenfoque σ=2": return np.stack([cv2.GaussianBlur(i, (0, 0), 2) for i in imgs])
    return imgs
out = []
for kind in ["original", "brillo -40%", "ruido σ=15", "desenfoque σ=2"]:
    imgs = degrade(images[te], kind); preds = m.predict(imgs, top_k=1)
    acc = np.mean([p.obj_id == o for p, o in zip(preds, index.obj_id.iloc[te])])
    out.append({"condición": kind, "acierto de clase": round(acc, 3), "confianza media": round(np.mean([p.class_prob for p in preds]), 3)})
pd.DataFrame(out)
""")
md("""
El control de calidad (desenfoque, exposición, fracción de objeto) existe porque el modelo pierde acierto con
condiciones que no vio: esas imágenes se desvían a revisión antes de llegar a la política.
""")
md("## 9. Exportación del artefacto v1.0.0")
code("""
subprocess.run([sys.executable, "-m", "src.vision.sweep", "--stage", "export"], check=True)
man = json.loads(Path("models/vision/v1.0.0/manifest.json").read_text(encoding="utf-8"))
print(json.dumps({k: man[k] for k in ["version", "data_sha256", "train_split", "unknown_objs", "model_sha256"]}, indent=1, ensure_ascii=False))
print(json.dumps(json.loads(Path("models/vision/v1.0.0/metrics.json").read_text()), indent=1)[:1500])
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("01_experimentacion.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("ok")
