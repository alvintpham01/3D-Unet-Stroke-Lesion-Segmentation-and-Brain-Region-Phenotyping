@torch.no_grad()
def run_real_case_wholebrain(t1_path, real_mask_path, title="", show_thr=0.8, show_overlay=True):
    print("Real case:")
    print("T1  :", t1_path)
    print("Mask:", real_mask_path)

    x_raw,_ = load_nii(t1_path)
    m_raw,_ = load_nii(real_mask_path)

    gt = (m_raw > 0.5).astype(np.float32)

    # --- whole-brain preprocessing (match training) ---
    x_np, y_np = downsample_vol_and_mask(x_raw.astype(np.float32), gt, ds_factor=DS_FACTOR)

    x_np = zscore_robust(x_np)  # or zscore_within_mask if that's what training used
    x_np = center_pad_or_crop_3d(x_np, TARGET_SHAPE, pad_value=0.0)
    y_np = center_pad_or_crop_3d(y_np, TARGET_SHAPE, pad_value=0.0)
    y_np = (y_np > 0.5).astype(np.float32)

    # --- forward ---
    x_t = torch.from_numpy(x_np[None, None]).float().to(device)
    y_t = torch.from_numpy(y_np[None, None]).float().to(device)

    logits = model(x_t)
    probs  = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
    gt_np  = y_np
    pred_show = (probs >= show_thr).astype(np.float32)

    # --- threshold sweep (optional, but useful) ---
    print("\n--- Threshold sweep ---")
    for thr in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8]:
        pred_thr = (probs >= thr).astype(np.float32)
        d = dice_coeff_bin(torch.from_numpy(pred_thr), torch.from_numpy(gt_np))
        print(f"thr={thr:.2f}  predvox={int(pred_thr.sum())}  dice={d:.4f}")

    # --- pick slice with most lesion (axial) ---
    # Assuming array is [D,H,W] and you want axial slices along D
    # --- pick slice with most lesion (AXIAL / transverse) ---
    z = int(np.argmax(gt_np.sum(axis=(0,1)))) if gt_np.sum() > 0 else (gt_np.shape[2]//2)

    import matplotlib.pyplot as plt
    ncols = 5 if show_overlay else 4
    plt.figure(figsize=(5*ncols, 5))

    plt.subplot(1,ncols,1); plt.imshow(x_np[:, :, z], cmap="gray"); plt.title("Input (axial)"); plt.axis("off")
    plt.subplot(1,ncols,2); plt.imshow(gt_np[:, :, z], cmap="gray"); plt.title("GT"); plt.axis("off")
    plt.subplot(1,ncols,3); plt.imshow(probs[:, :, z], cmap="magma"); plt.title("Prob map"); plt.axis("off")
    plt.subplot(1,ncols,4); plt.imshow(pred_show[:, :, z], cmap="gray"); plt.title(f"Pred @thr={show_thr:.2f}"); plt.axis("off")

    if show_overlay:
        plt.subplot(1,ncols,5)
        plt.imshow(x_np[:, :, z], cmap="gray")
        plt.imshow(pred_show[:, :, z], cmap="Reds", alpha=0.35)
        plt.title("Pred overlay")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

        # ---- EXTRA: Overlay Input + GT + Pred@0.5 ----
    pred_08 = (probs >= 0.8).astype(np.uint8)

    base = x_np[:, :, z].astype(np.float32)
    # normalize base to [0,1] for display
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)

    gt2   = (gt_np[:, :, z] > 0.5).astype(np.uint8)
    pred2 = (pred_08[:, :, z] > 0.5).astype(np.uint8)

    # start from grayscale RGB
    rgb = np.stack([base, base, base], axis=-1)

    # color masks
    alpha = 0.55  # increase for stronger color

    # red for pred
    rgb[..., 0] = np.clip(rgb[..., 0] + alpha * pred2, 0, 1)
    # green for gt
    rgb[..., 1] = np.clip(rgb[..., 1] + alpha * gt2, 0, 1)

    # (optional) make overlap pop as yellow by boosting both channels a bit more
    overlap = (gt2 & pred2).astype(np.uint8)
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.25 * overlap, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.25 * overlap, 0, 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.title("Overlay: GT (green) + Pred@0.15 (red); overlap=yellow")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
