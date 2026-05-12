
@torch.no_grad()
def eval_dataset_wholevol(pairs, thr=0.3, max_cases=None):
    model.eval()
    rows=[]
    dices=[]
    for i,(t1_path,m_path) in enumerate(pairs):
        if max_cases is not None and i>=max_cases: break
        x_np,_ = load_nii(t1_path)
        y_np,_ = load_nii(m_path)
        y_np = (y_np > 0.5).astype(np.float32)

        x_np, y_np = downsample_vol_and_mask(x_np, y_np, ds_factor=DS_FACTOR)
        x_np = zscore_robust(x_np)
        x_np = center_pad_or_crop_3d(x_np, TARGET_SHAPE, pad_value=0.0)
        y_np = center_pad_or_crop_3d(y_np, TARGET_SHAPE, pad_value=0.0)
        y_np = (y_np > 0.5).astype(np.float32)

        x_t = torch.from_numpy(x_np[None,None]).float().to(device)
        y_t = torch.from_numpy(y_np[None,None]).float().to(device)

        logits = model(x_t)
        probs = torch.sigmoid(logits)
        pred = (probs >= thr).float()
        d = dice_coeff_bin(pred, y_t)

        gtvox = float(y_t.sum().item())
        predvox = float(pred.sum().item())
        rows.append((i, os.path.basename(t1_path), d, gtvox, predvox))
        dices.append(d)

    return rows, dices

ths = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8]

mean_dices = []
median_dices = []

for thr in ths:
    rows, dices = eval_dataset_wholevol(test_pairs, thr=thr, max_cases=None)
    mean_dices.append(np.mean(dices))
    median_dices.append(np.median(dices))
    print(f"thr={thr:.2f}  dice_mean={np.mean(dices):.4f}  dice_median={np.median(dices):.4f}")



#FIGRURE

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(ths, mean_dices, marker='o', color='royalblue', label="Mean Dice")
plt.plot(ths, median_dices, marker='s', color='#BF5700', label="Median Dice")
plt.xlabel("Threshold")
plt.ylabel("Dice")
plt.title("Dice vs Threshold")
plt.grid(False)
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
