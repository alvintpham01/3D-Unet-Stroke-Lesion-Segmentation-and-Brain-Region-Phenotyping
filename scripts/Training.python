#8) Training loop + validation Dice on **whole volumes** 
base_ch = 32
model = UNet3D(in_ch=1, base=base_ch, p_drop=0.0).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

FAST_PREVIEW = False
OPT_STEPS = 200 if FAST_PREVIEW else 20000 # optimizer updates
ACCUM_STEPS = 4 if device.type=='cuda' else 1  # effective batch size = ACCUM_STEPS

LOG_EVERY = 10
VAL_EVERY = 100



def dice_coeff_bin(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2*inter + eps) / (denom + eps))

@torch.no_grad()
def eval_val_wholevol(model, loader, thr=0.3, max_cases=None):
    model.eval()
    dices=[]
    for i,(x_t,y_t) in enumerate(loader):
        if max_cases is not None and i>=max_cases: break
        x_t = x_t.to(device, non_blocking=True)
        y_t = y_t.to(device, non_blocking=True)
        logits = model(x_t)
        probs = torch.sigmoid(logits)
        pred = (probs >= thr).float()
        d = dice_coeff_bin(pred, y_t)
        dices.append(d)
    return float(np.mean(dices)) if dices else 0.0

def save_ckpt(path, model, opt, scaler, step):
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": int(step)}, path)


# =========================
# 8) Training loop (whole volume) + validation
# =========================
# Lighter model is usually needed for whole-volume training
# If you OOM, reduce base_ch to 16 (or even 8).

start_step = 0

model.train()
it = iter(train_loader)
running = 0.0
opt.zero_grad(set_to_none=True)
####################################
for step in range(start_step + 1, start_step + OPT_STEPS + 1):
    try:
        x_t, y_t = next(it)
    except StopIteration:
        it = iter(train_loader)
        x_t, y_t = next(it)

    x_t = x_t.to(device, non_blocking=True)
    y_t = y_t.to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        logits = model(x_t)
        loss = combo_loss(logits, y_t)

        # normalize for accumulation
        loss = loss / ACCUM_STEPS

    scaler.scale(loss).backward()
    running += float(loss.item())

    if step % ACCUM_STEPS == 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    if step % LOG_EVERY == 0:
        print(f"{step:04d}  loss={running/LOG_EVERY:.4f}")
        running = 0.0

    if step % VAL_EVERY == 0:
         d = eval_val_wholevol(model, val_loader, thr=0.3, max_cases=8)
         print(f"   [val @thr=0.30] dice={d:.4f}")
         save_ckpt("checkpoint_latest_posw10_full.pth", model, opt, scaler, step)
         print("   Saved checkpoint_latest_posw10_full.pth")

# save
torch.save(model.state_dict(), OUT_CKPT)
print("Saved:", OUT_CKPT)
