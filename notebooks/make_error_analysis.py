"""Generate the error-analysis notebook for a trained experiment.

    python notebooks/make_error_analysis.py improved_all101 [--notes notes.md]
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_error_analysis.ipynb
"""
import sys, nbformat as nbf
import argparse, os
ap = argparse.ArgumentParser(description="Generate notebooks/02_error_analysis.ipynb for an experiment (then execute it with nbconvert).")
ap.add_argument("experiment", help="e.g. improved_all101 or full_convnext_tiny")
ap.add_argument("--notes", help="markdown file with the written failure analysis")
args = ap.parse_args()
EXP = args.experiment
NOTES = open(args.notes).read() if args.notes else "_Notes to be written after reviewing the outputs above._"
nb = nbf.v4.new_notebook(); M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
M(f"# 02 · Error analysis: `{EXP}`\n\nInputs come from `python -m src.evaluate` (`results/{EXP}/`). Goal: understand *which* dishes fail, *why*, and *what a mistake costs* in calories."),
C(f"""import os, sys, json
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from PIL import Image
ROOT = os.path.abspath('..'); EXP = '{EXP}'
R = f'{{ROOT}}/results/{{EXP}}'
IMG_ROOT = f'{{ROOT}}/data/cache/food101_256'
metrics = json.load(open(f'{{R}}/metrics.json')); metrics"""),
M("## Per-class F1: hardest and easiest dishes"),
C("""pc = pd.read_csv(f'{R}/per_class.csv').sort_values('f1')
show = pd.concat([pc.head(12), pc.tail(8)])
plt.figure(figsize=(8, 6))
plt.barh(show['class'].str.replace('_',' '), show.f1, color=['#e76f51']*12 + ['#2a9d8f']*8)
plt.xlabel('F1 (test)'); plt.title('12 hardest vs 8 easiest classes'); plt.xlim(0, 1); plt.tight_layout()
plt.savefig(f'{ROOT}/assets/per_class_f1.png', dpi=100); plt.show()
pc.head(12)"""),
M("## Most-confused pairs (true → predicted)"),
C("""cp = pd.read_csv(f'{R}/confused_pairs.csv').head(15)
lab = cp.true.str.replace('_',' ') + '  →  ' + cp.predicted.str.replace('_',' ')
plt.figure(figsize=(8, 5)); plt.barh(lab[::-1], cp['count'][::-1], color='#264653')
plt.xlabel('test images'); plt.title('Top-15 confusions'); plt.tight_layout()
plt.savefig(f'{ROOT}/assets/confused_pairs.png', dpi=100); plt.show()
cp"""),
M("## Confidence: is the model's confidence meaningful?\nThe app warns when top-1 confidence < 0.4. This checks that low confidence really does flag likely errors, and how much a user-correction step recovers (true label inside the top-3 shown to the user)."),
C("""pr = pd.read_csv(f'{R}/predictions.csv')
pr['top3'] = pr.top5.str.split('|').apply(lambda xs: [x.split(':')[0] for x in xs[:3]])
pr['in_top3'] = [t in s for t, s in zip(pr.true, pr.top3)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist([pr.conf[pr.correct == 1], pr.conf[pr.correct == 0]], bins=20, label=['correct', 'wrong'], color=['#2a9d8f', '#e76f51'])
ax[0].axvline(0.4, ls='--', c='k'); ax[0].set(title='Top-1 confidence', xlabel='confidence'); ax[0].legend()
th = np.linspace(0, 0.95, 40)
ax[1].plot(th, [pr.correct[pr.conf >= t].mean() for t in th], label='accuracy on kept images')
ax[1].plot(th, [(pr.conf >= t).mean() for t in th], label='coverage (fraction kept)')
ax[1].axvline(0.4, ls='--', c='k'); ax[1].set(title='Accuracy vs. confidence threshold', xlabel='threshold', ylim=(0, 1.02)); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{ROOT}/assets/confidence.png', dpi=100); plt.show()
low = pr.conf < 0.4
print(f"top-1 {pr.correct.mean():.3f} | top-3 {pr.in_top3.mean():.3f}")
print(f"images flagged low-confidence: {low.mean():.1%}; accuracy flagged {pr.correct[low].mean():.3f} vs not flagged {pr.correct[~low].mean():.3f}")
print(f"errors recoverable by picking from the top-3 list: {pr.in_top3[pr.correct == 0].mean():.1%}")"""),
M("## What does a mistake cost in calories?\nFor every misclassified image: |kcal(predicted dish) − kcal(true dish)| for one typical serving of the true dish's weight. Confusing two similar cakes is cheap; confusing a salad with a cake is not."),
C("""from src.nutrition import load_db
db = load_db()
err = pr[pr.correct == 0].copy()
err['kcal_true'] = [db.estimate(t, db.get(t).serving_g).kcal for t in err.true]
err['kcal_pred'] = [db.estimate(p, db.get(t).serving_g).kcal for p, t in zip(err.pred, err.true)]
err['abs_err'] = (err.kcal_pred - err.kcal_true).abs()
err['rel_err'] = err.abs_err / err.kcal_true
print(err[['abs_err', 'rel_err']].describe(percentiles=[.5, .9]).round(2))
print(f"\\nmisclassifications costing <20% kcal error: {(err.rel_err < 0.2).mean():.1%}")
worst = err.groupby(['true', 'pred']).agg(n=('abs_err', 'size'), kcal_err=('abs_err', 'mean')).query('n >= 2').sort_values('kcal_err', ascending=False).head(10)
worst.round(0)"""),
M("## Failure gallery: the most confident mistakes\nEach pair: the photo with Grad-CAM for the **wrong** predicted class, then Grad-CAM for the **true** class."),
C("""from src.predict import Predictor
ckpt = f'{ROOT}/checkpoints/{EXP}/best.pt'
P = Predictor(ckpt, tta=False)
wrong = pr[pr.correct == 0].sort_values('conf', ascending=False).head(8)
fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
for (_, r), ax2 in zip(wrong.iterrows(), axes.reshape(-1, 2)):
    img = Image.open(f'{IMG_ROOT}/{r.path}').convert('RGB')
    ax2[0].imshow(P.explain(img, r.pred)); ax2[0].set_title(f'pred: {r.pred}\\n({r.conf:.2f})', fontsize=9, color='#c0392b')
    ax2[1].imshow(P.explain(img, r.true)); ax2[1].set_title(f'true: {r.true}', fontsize=9, color='#1e8449')
    for a in ax2: a.axis('off')
plt.tight_layout(); plt.savefig(f'{ROOT}/assets/failure_cases.jpg', dpi=80); plt.show()"""),
M("## Notes: why these fail\n\n" + NOTES),
]
nb.metadata["kernelspec"] = {"name": "nutrisnap", "display_name": "nutrisnap", "language": "python"}
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_error_analysis.ipynb"))
print("wrote notebooks/02_error_analysis.ipynb")
