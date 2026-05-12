# =========================
# 5) Whole-volume dataset + fixed shape
# =========================

CONFIG = {
    "ds_factor": 1,
    "augment": True,
    "base_ch": 16,
    "dropout": 0.1,
    "pos_weight": 20,
    "dice_weight": 0.8,
    "bce_weight": 0.2,
}

DS_FACTOR = CONFIG["ds_factor"]
AUGMENT = CONFIG["augment"]


def round_up_to_multiple(x, m=16):
    return int(math.ceil(x / m) * m)


@torch.no_grad()
def downsample_vol_and_mask(x_np, y_np, ds_factor=1):
    assert ds_factor >= 1 and int(ds_factor) == ds_factor

    if ds_factor == 1:
        return x_np.astype(np.float32), y_np.astype(np.float32)

    x = torch.from_numpy(x_np[None, None].astype(np.float32))
    y = torch.from_numpy(y_np[None, None].astype(np.float32))

    D, H, W = x.shape[-3:]
    D2 = max(1, D // ds_factor)
    H2 = max(1, H // ds_factor)
    W2 = max(1, W // ds_factor)

    x2 = F.interpolate(
        x,
        size=(D2, H2, W2),
        mode="trilinear",
        align_corners=False,
    )

    y2 = F.interpolate(
        y,
        size=(D2, H2, W2),
        mode="nearest",
    )

    return (
        x2[0, 0].numpy().astype(np.float32),
        y2[0, 0].numpy().astype(np.float32),
    )


def center_pad_or_crop_3d(x_np, target_shape, pad_value=0.0):
    D, H, W = x_np.shape
    TD, TH, TW = target_shape

    # Center crop
    sd = max(0, (D - TD) // 2)
    sh = max(0, (H - TH) // 2)
    sw = max(0, (W - TW) // 2)

    x = x_np[
        sd:sd + min(D, TD),
        sh:sh + min(H, TH),
        sw:sw + min(W, TW),
    ]

    # Center pad
    D2, H2, W2 = x.shape

    pd0 = max(0, (TD - D2) // 2)
    pd1 = max(0, TD - D2 - pd0)

    ph0 = max(0, (TH - H2) // 2)
    ph1 = max(0, TH - H2 - ph0)

    pw0 = max(0, (TW - W2) // 2)
    pw1 = max(0, TW - W2 - pw0)

    if pd0 or pd1 or ph0 or ph1 or pw0 or pw1:
        x = np.pad(
            x,
            ((pd0, pd1), (ph0, ph1), (pw0, pw1)),
            mode="constant",
            constant_values=pad_value,
        )

    return x.astype(np.float32)


def compute_target_shape(pairs, ds_factor=1, max_check=24):
    maxD, maxH, maxW = 0, 0, 0

    scan = pairs[:max_check] if len(pairs) > max_check else pairs

    for t1_path, m_path in scan:
        x, _ = load_nii(t1_path)
        y, _ = load_nii(m_path)

        y = (y > 0.5).astype(np.float32)

        x2, y2 = downsample_vol_and_mask(
            x,
            y,
            ds_factor=ds_factor,
        )

        D, H, W = x2.shape

        maxD = max(maxD, D)
        maxH = max(maxH, H)
        maxW = max(maxW, W)

    return (
        round_up_to_multiple(maxD, 16),
        round_up_to_multiple(maxH, 16),
        round_up_to_multiple(maxW, 16),
    )


TARGET_SHAPE = compute_target_shape(
    train_pairs,
    ds_factor=DS_FACTOR,
    max_check=24,
)

print("DS_FACTOR:", DS_FACTOR)
print("TARGET_SHAPE:", TARGET_SHAPE)


class AtlasSegWholeVolDataset(Dataset):
    def __init__(
        self,
        pairs,
        ds_factor=1,
        target_shape=(208, 240, 192),
        augment=True,
    ):
        self.pairs = pairs
        self.ds = ds_factor
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t1_path, m_path = self.pairs[idx]

        x_np, _ = load_nii(t1_path)
        y_np, _ = load_nii(m_path)

        y_np = (y_np > 0.5).astype(np.float32)

        x_np, y_np = downsample_vol_and_mask(
            x_np,
            y_np,
            ds_factor=self.ds,
        )

        brain_mask = brain_mask_robust_raw(x_np)
        x_np = zscore_within_mask(x_np, brain_mask)

        x_np = center_pad_or_crop_3d(
            x_np,
            self.target_shape,
            pad_value=0.0,
        )

        y_np = center_pad_or_crop_3d(
            y_np,
            self.target_shape,
            pad_value=0.0,
        )

        y_np = (y_np > 0.5).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                x_np = x_np[::-1, :, :].copy()
                y_np = y_np[::-1, :, :].copy()

            if random.random() < 0.5:
                x_np = x_np[:, ::-1, :].copy()
                y_np = y_np[:, ::-1, :].copy()

            if random.random() < 0.5:
                x_np = x_np[:, :, ::-1].copy()
                y_np = y_np[:, :, ::-1].copy()

        x_t = torch.from_numpy(x_np[None]).float()
        y_t = torch.from_numpy(y_np[None]).float()

        return x_t, y_t


train_ds = AtlasSegWholeVolDataset(
    train_pairs,
    ds_factor=DS_FACTOR,
    target_shape=TARGET_SHAPE,
    augment=AUGMENT,
)

val_ds = AtlasSegWholeVolDataset(
    val_pairs,
    ds_factor=DS_FACTOR,
    target_shape=TARGET_SHAPE,
    augment=False,
)

test_ds = AtlasSegWholeVolDataset(
    test_pairs,
    ds_factor=DS_FACTOR,
    target_shape=TARGET_SHAPE,
    augment=False,
)

train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# =========================
# 6) 3D U-Net
# =========================

def center_crop_to(x, ref):
    _, _, d, h, w = x.shape
    _, _, rd, rh, rw = ref.shape

    assert d >= rd and h >= rh and w >= rw

    sd = (d - rd) // 2
    sh = (h - rh) // 2
    sw = (w - rw) // 2

    return x[:, :, sd:sd + rd, sh:sh + rh, sw:sw + rw]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, base=16, p_drop=0.1):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base, p_drop)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base, base * 2, p_drop)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(base * 2, base * 4, p_drop)
        self.pool3 = nn.MaxPool3d(2)

        self.bott = DoubleConv(base * 4, base * 8, p_drop)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4, p_drop)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2, p_drop)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base, p_drop)

        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bott(self.pool3(e3))

        d3 = self.up3(b)

        if e3.shape[2:] != d3.shape[2:]:
            e3 = center_crop_to(e3, d3)

        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)

        if e2.shape[2:] != d2.shape[2:]:
            e2 = center_crop_to(e2, d2)

        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)

        if e1.shape[2:] != d1.shape[2:]:
            e1 = center_crop_to(e1, d1)

        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


model = UNet3D(
    in_ch=1,
    base=CONFIG["base_ch"],
    p_drop=CONFIG["dropout"],
).to(device)

print(model.__class__.__name__)


# =========================
# 7) Losses
# =========================

pos_w = torch.tensor(
    [CONFIG["pos_weight"]],
    device=device,
)

bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)


def soft_dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


def combo_loss(
    logits,
    targets,
    w_dice=CONFIG["dice_weight"],
    w_bce=CONFIG["bce_weight"],
):
    return (
        w_dice * soft_dice_loss_from_logits(logits, targets)
        + w_bce * bce(logits, targets)
    )
