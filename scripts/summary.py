import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import random
sns.set(style="whitegrid")
import time

if not os.path.exists("data/plots"):
    os.mkdir("data/plots")
# model_names = ["mla_sigma0.5","mla_sigma0.8","mla_sigma1.0"]
# model_names = ["mla_drop0.0","mla_drop0.15","mla_drop0.3","mla_drop0.45","mla_drop0.6"]
# model_names = ["mla_type0","mla_type1","mla_type2","mla_type3","mla_nrsub", "mla_fgsub"]
# model_names = ["mla_da"]
# model_names = ["mla_nrsub", "mla_fgsub"]"]
# model_names = ["mla_aug_50d","mla_aug_back","mla_aug3","mla_aug2"]
# model_names = ["mla_50d","mla","mla_type0"]

# model_names = ["mla_aug3","mla_aug_back","mla_aug2"]
model_names = [
    # "mla_bert",
    # "mla_bert_new",
    # "mla_bert_cls",
    # "mla_bert_use2",
    # "mla_bert_use4",
    # "transformer",
    # "transformer_lstm",
    # "transformer_multidec",
    # "transformer_dropin",
    # "transformer_p0_tune",
    "transformerv5_p2_tune2_1",
    # "transformer_p2_tune_old",
    # "transformer_none",
    # "transformer_token",
    # "transformer_split",
    # "transformer_pt",
    # "transformer_rgb3",
    # "transformer_rgb4",
    # "transformer_preln",
    # "transformer_nopostln",
    # "transformer_moregru",
    # "transformer_traindepth",
    # "transformer_add"
    # "mla_fgsub_highratio",
]
model_prefix = "transformer_"
checkpoint_path = os.path.join("data", "checkpoints")
eval_folder = "evals"
s = "unseen"
val_set = "val_%s" % (s)
summary_file = "summary_%s_base.json" % (s)
summary = {}
metrics = []
for model_name in model_names:
    print(model_name)
    result = {}
    result_files = list(
        os.listdir(os.path.join(checkpoint_path, model_name, eval_folder))
    )
    for result_file in result_files:
        print(result_file)
        if val_set in result_file and "json" in result_file:
            path = os.path.join(
                checkpoint_path, model_name, eval_folder, result_file
            )
            with open(path, "r") as f:
                data = json.loads(f.read())
            idx = int(result_file.split("_")[2])
            result[idx] = data
    result = sorted(result.items(), key=lambda d: d[0], reverse=False)
    index = [int(v[0]) for v in result]
    result = [v[1] for v in result]
    try:
        metrics = list(result[0].keys())
        result = dict(
            zip(
                list(result[0].keys()),
                list(zip(*[[k[1] for k in v.items()] for v in result])),
            )
        )
        for k, v in result.items():
            result[k] = dict(zip(index, v))
        summary[model_name] = result
    except IndexError:
        pass
with open(os.path.join(checkpoint_path, summary_file), "w") as f:
    f.write(json.dumps(summary))

with open(os.path.join(checkpoint_path, summary_file), "r") as f:
    summary = json.loads(f.read())
metrics = ["success", "spl"]

fig, ax = plt.subplots(len(metrics), 1, figsize=(12.8, 6.4 * len(metrics)))
clist = ['aqua',
 'black',
 'blue',
 'brown',
 'darkcyan',
 'darkgreen',
 'darkmagenta',
 'darkorchid',
 'darkred',
 'darkslategray',
 'darkviolet',
 'deeppink',
 'fuchsia',
 'indigo',
 'lime',
 'magenta',
 'maroon',
 'navy',
 'orangered']
markers = ["o", "+", "x", "*", "D","v","^","<",">","s","p","h","X","D"]
print("%-16s | %10s | %10s | %10s"%("model_prefix","metric","mean","max"))
for i, metric in enumerate(metrics):
    # fig, ax = plt.subplots()
    for j, (k, v) in enumerate(summary.items()):
        now = sorted([(int(z), t) for z, t in v[metric].items()])
        now = dict(now)
        x = list(now.keys())
        y = list(now.values())
        ax[i].plot(x, y, label=k,markersize=10, marker=markers[j])
        val = np.array(list(y))
        print("%-16s | %10s | %10.4f | %10.4f"%(k.replace(model_prefix,""),metric,np.mean(val),np.max(val)))
    # .legend()
    ax[i].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.,fontsize=14)     ##设置ax4中legend的位置，将其放在图外
    ax[i].set_title(val_set + "_" + metric, fontsize=20)
    ax[i].tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(
    os.path.join("data", "plots", "%s_%d.jpg" % (val_set, time.time())),
    dpi=300,
)

# with open(os.path.join(checkpoint_path, summary_file), "r") as f:
#     summary = json.loads(f.read())
# del summary["cma_pm_aug"]
# for metric in ["success", "spl"]:
#     fig, ax = plt.subplots(2, 2)
#     for i, (k, v) in enumerate(summary.items()):
#         ax[int(i/2), int(i%2)].plot(v[metric], label=k)
#         ax[int(i/2), int(i%2)].set_title(k)
#     fig.suptitle(val_set+"_"+metric)
#     fig.tight_layout()
#     fig.savefig(os.path.join("data", "plots", "plots_success", "%s_%s.jpg"%(val_set, metric)))
