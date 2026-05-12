import os
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# 1. USER SETTINGS
# ============================================================

D1_PATH = "D1.csv"
D2_PATH = "D2.csv"
D3_PATH = "D3.csv"

OUTPUT_DIR = "dt_gcil_final_pubquality_v4_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "ID",
    "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7"
]

LABEL_COLS = ["label", "category", "specific_class"]

RANDOM_STATE = 42
TEST_SIZE_D1 = 0.25
TEST_SIZE_NEW = 0.50

BATCH_SIZE = 256
BASE_EPOCHS = 15
CIL_EPOCHS = 8
FULL_FL_EPOCHS = 15
LEARNING_RATE = 1e-3

REPLAY_PER_CLASS = 1000
LAMBDA_DISTILL = 0.7
LAMBDA_REG = 1e-4

GAN_SYNTHETIC_PER_CLASS = 1000
GAN_NOISE_SCALE = 0.05

UNTRUSTED_CORRUPTION_RATE = 0.25
UNTRUSTED_FEATURE_NOISE = 0.15

MAX_ROWS_PER_DATASET = None
# If slow:
# MAX_ROWS_PER_DATASET = 50000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIME_MIN = 350
ATTACK_START_MIN = 120
ATTACK_END_MIN = 300
SAMPLES_PER_MIN = 20
ATTACK_RATIO_DURING_ATTACK = 0.7
WINDOW_SIZE_MIN = 10

# Communication assumptions
N_AV = 5
TRUSTED_FRACTION = 0.6
FLOAT_BYTES = 4


# ============================================================
# 2. METHOD NAMES AND COLORS
# ============================================================

METHOD_FULL = "Full FL"
METHOD_REG = "Reg-CIL"
METHOD_DTG = "DT-GCIL"
METHOD_DTG_NOGAN = "DT-GCIL w/o GAN"
METHOD_DTG_NOTRUST = "DT-GCIL w/o Trust"

METHOD_ORDER = [METHOD_FULL, METHOD_REG, METHOD_DTG]

MODEL_COLORS = {
    METHOD_FULL: "black",
    METHOD_REG: "blue",
    METHOD_DTG: "red",
    METHOD_DTG_NOGAN: "blue",
    METHOD_DTG_NOTRUST: "orange"
}

MODEL_MARKERS = {
    METHOD_FULL: "o",
    METHOD_REG: "s",
    METHOD_DTG: "^",
    METHOD_DTG_NOGAN: "D",
    METHOD_DTG_NOTRUST: "v"
}


# ============================================================
# 3. FIGURE STYLE
# ============================================================

def set_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 800,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.size": 8,
        "font.weight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,

        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.22,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.35,

        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,

        "grid.linewidth": 0.45,
        "grid.alpha": 0.30,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "ps.fonttype": 42,
        "pdf.fonttype": 42
    })


def make_axes_bold(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.tick_params(axis="both", which="major", width=1.0, length=3.5)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def save_figure_both(fig, base_path_no_ext):
    fig.savefig(base_path_no_ext + ".png", format="png", dpi=800, bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".eps", format="eps", bbox_inches="tight")


def compact_legend(ax, loc="upper center", ncol=3):
    ax.legend(
        loc=loc,
        ncol=ncol,
        fontsize=7,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        handlelength=1.2,
        columnspacing=0.55,
        borderpad=0.25,
        labelspacing=0.22
    )


set_publication_style()


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


set_seed(RANDOM_STATE)


# ============================================================
# 5. DATA LOADING
# ============================================================

def hex_or_number_to_float(x):
    if pd.isna(x):
        return 0.0

    x = str(x).strip()

    if x == "":
        return 0.0

    try:
        if x.lower().startswith("0x"):
            return float(int(x, 16))

        if any(c in x.upper() for c in ["A", "B", "C", "D", "E", "F"]):
            return float(int(x, 16))

        return float(x)

    except Exception:
        return 0.0


def load_dataset(path, max_rows=None):
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS + LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    for col in FEATURE_COLS:
        df[col] = df[col].apply(hex_or_number_to_float)

    df["target"] = (
        df["label"].astype(str)
        + "_"
        + df["category"].astype(str)
        + "_"
        + df["specific_class"].astype(str)
    )

    return df


def safe_split(df, test_size):
    counts = df["target"].value_counts()
    stratify = df["target"] if len(counts) > 1 and counts.min() >= 2 else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_replay_buffer(df, replay_per_class):
    parts = []

    for _, group in df.groupby("target"):
        n = min(len(group), replay_per_class)
        parts.append(group.sample(n, random_state=RANDOM_STATE))

    return pd.concat(parts).reset_index(drop=True)


# ============================================================
# 6. GAN-LIKE REPLAY AND TRUST SIMULATION
# ============================================================

def generate_gan_like_replay(replay_df, synthetic_per_class=GAN_SYNTHETIC_PER_CLASS):
    synthetic_parts = []

    for target, group in replay_df.groupby("target"):
        n = min(synthetic_per_class, max(1, len(group)))

        numeric = group[FEATURE_COLS].astype(float)
        mu = numeric.mean().values
        sigma = numeric.std().replace(0, 1e-6).values

        synth_x = np.random.normal(
            loc=mu,
            scale=GAN_NOISE_SCALE * sigma,
            size=(n, len(FEATURE_COLS))
        )

        synth_df = pd.DataFrame(synth_x, columns=FEATURE_COLS)

        rep = group.iloc[0]
        synth_df["label"] = rep["label"]
        synth_df["category"] = rep["category"]
        synth_df["specific_class"] = rep["specific_class"]
        synth_df["target"] = target

        synthetic_parts.append(synth_df)

    if len(synthetic_parts) == 0:
        return replay_df.iloc[:0].copy()

    return pd.concat(synthetic_parts, axis=0).reset_index(drop=True)


def corrupt_untrusted_updates(train_df, class_values, corruption_rate=UNTRUSTED_CORRUPTION_RATE):
    if corruption_rate <= 0:
        return train_df

    df = train_df.copy().reset_index(drop=True)
    n = len(df)
    n_corrupt = int(round(n * corruption_rate))

    if n_corrupt <= 0:
        return df

    rng = np.random.default_rng(RANDOM_STATE)
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)

    class_values = list(class_values)

    for idx in corrupt_idx:
        current = df.loc[idx, "target"]
        candidates = [c for c in class_values if c != current]

        if len(candidates) > 0:
            df.loc[idx, "target"] = rng.choice(candidates)

    feature_std = df[FEATURE_COLS].astype(float).std().replace(0, 1.0).values

    noise = rng.normal(
        0,
        UNTRUSTED_FEATURE_NOISE * feature_std,
        size=(n_corrupt, len(FEATURE_COLS))
    )

    df.loc[corrupt_idx, FEATURE_COLS] = df.loc[corrupt_idx, FEATURE_COLS].astype(float).values + noise

    return df


# ============================================================
# 7. MODEL
# ============================================================

class ExpandableIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=32, num_classes=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

    def expand_classes(self, n_new):
        old_head = self.classifier
        old_out = old_head.out_features
        in_features = old_head.in_features

        new_head = nn.Linear(in_features, old_out + n_new)

        with torch.no_grad():
            new_head.weight[:old_out] = old_head.weight
            new_head.bias[:old_out] = old_head.bias
            nn.init.xavier_uniform_(new_head.weight[old_out:])
            nn.init.zeros_(new_head.bias[old_out:])

        self.classifier = new_head.to(DEVICE)


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def classifier_head_params(feature_dim, num_classes):
    return (feature_dim + 1) * num_classes


def dtgcil_increment_params(feature_dim, n_new_classes):
    # New head weights + bias + prototype vector + one trust score
    return n_new_classes * ((feature_dim + 1) + feature_dim + 1)


# ============================================================
# 8. HELPERS
# ============================================================

def build_class_order(D1, D2, D3):
    d1_classes = sorted(D1["target"].unique().tolist())
    d2_classes = sorted([c for c in D2["target"].unique() if c not in d1_classes])
    d3_classes = sorted([c for c in D3["target"].unique() if c not in d1_classes + d2_classes])
    return d1_classes + d2_classes + d3_classes


def df_to_tensor(df, scaler, class_to_idx):
    X = scaler.transform(df[FEATURE_COLS]).astype(np.float32)
    y = df["target"].map(class_to_idx).values.astype(np.int64)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    return X_t, y_t


def make_loader(df, scaler, class_to_idx, batch_size=BATCH_SIZE, shuffle=True):
    X_t, y_t = df_to_tensor(df, scaler, class_to_idx)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict_label_names(model, df, scaler, class_to_idx, batch_size=4096):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    X = scaler.transform(df[FEATURE_COLS]).astype(np.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    preds_all = []

    model.eval()
    sync_device()
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size].to(DEVICE)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.append(preds)

    sync_device()
    elapsed = time.perf_counter() - start

    preds_all = np.concatenate(preds_all, axis=0)
    pred_labels = np.array([idx_to_class[p] for p in preds_all])
    true_labels = df["target"].values

    return true_labels, pred_labels, elapsed


def evaluate_multiclass(model, df, scaler, class_to_idx):
    true_labels, pred_labels, test_time = predict_label_names(
        model,
        df,
        scaler,
        class_to_idx
    )

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    return {
        "accuracy_percent": acc * 100,
        "macro_f1_percent": macro_f1 * 100,
        "testing_time_sec": test_time,
        "n_samples": len(df),
        "testing_time_ms_per_sample": (test_time * 1000.0) / len(df)
    }, true_labels, pred_labels


def evaluate_task_accuracy(model, task_df, scaler, class_to_idx):
    true_labels, pred_labels, test_time = predict_label_names(
        model,
        task_df,
        scaler,
        class_to_idx
    )

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    return {
        "accuracy_percent": acc * 100,
        "macro_f1_percent": macro_f1 * 100,
        "testing_time_sec": test_time
    }


# ============================================================
# 9. TRAINING FUNCTIONS
# ============================================================

def train_model_from_scratch(train_df, scaler, class_to_idx, num_classes, epochs):
    input_dim = len(FEATURE_COLS)

    model = ExpandableIDS(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(DEVICE)

    loader = make_loader(train_df, scaler, class_to_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sync_device()
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()

        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

    sync_device()
    elapsed = time.perf_counter() - start

    return model, elapsed


def regularization_loss(current_model, old_model):
    loss = 0.0

    current_params = dict(current_model.named_parameters())
    old_params = dict(old_model.named_parameters())

    for name, p_old in old_params.items():
        if name not in current_params:
            continue

        p_new = current_params[name]

        if p_new.shape == p_old.shape:
            loss += torch.sum((p_new - p_old) ** 2)
        else:
            slices = tuple(slice(0, s) for s in p_old.shape)
            loss += torch.sum((p_new[slices] - p_old) ** 2)

    return loss


def update_regularization_cil(model, new_df, scaler, class_to_idx):
    old_model = copy.deepcopy(model).to(DEVICE)
    old_model.eval()

    loader = make_loader(new_df, scaler, class_to_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sync_device()
    start = time.perf_counter()

    for _ in range(CIL_EPOCHS):
        model.train()

        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)
            ce_loss = F.cross_entropy(logits, y)
            reg_loss = regularization_loss(model, old_model)

            loss = ce_loss + LAMBDA_REG * reg_loss
            loss.backward()
            optimizer.step()

    sync_device()
    elapsed = time.perf_counter() - start

    return elapsed


def update_dtgcil(
    model,
    new_df,
    replay_df,
    scaler,
    class_to_idx,
    old_class_count,
    use_gan=True,
    trust_aware=True
):
    old_model = copy.deepcopy(model).to(DEVICE)
    old_model.eval()

    train_parts = [new_df, replay_df]

    if use_gan:
        train_parts.append(generate_gan_like_replay(replay_df))

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)

    if not trust_aware:
        seen_classes = train_df["target"].unique().tolist()
        train_df = corrupt_untrusted_updates(train_df, seen_classes)

    loader = make_loader(train_df, scaler, class_to_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    temperature = 2.0

    sync_device()
    start = time.perf_counter()

    for _ in range(CIL_EPOCHS):
        model.train()

        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)
            ce_loss = F.cross_entropy(logits, y)

            old_mask = y < old_class_count

            if old_mask.sum() > 0:
                X_old = X[old_mask]

                with torch.no_grad():
                    old_logits = old_model(X_old)[:, :old_class_count]

                new_logits_old = model(X_old)[:, :old_class_count]

                distill_loss = F.kl_div(
                    F.log_softmax(new_logits_old / temperature, dim=1),
                    F.softmax(old_logits / temperature, dim=1),
                    reduction="batchmean"
                ) * (temperature ** 2)

            else:
                distill_loss = torch.tensor(0.0, device=DEVICE)

            loss = ce_loss + LAMBDA_DISTILL * distill_loss
            loss.backward()
            optimizer.step()

    sync_device()
    elapsed = time.perf_counter() - start

    return elapsed


# ============================================================
# 10. METRICS
# ============================================================

def binary_attack_metrics(model, positive_df, negative_df, positive_class_name, scaler, class_to_idx):
    test_df = pd.concat([positive_df, negative_df], axis=0).reset_index(drop=True)

    true_labels, pred_labels, test_time = predict_label_names(
        model,
        test_df,
        scaler,
        class_to_idx
    )

    true_binary = (true_labels == positive_class_name).astype(int)
    pred_binary = (pred_labels == positive_class_name).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_binary,
        pred_binary,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy_percent": accuracy * 100,
        "Precision_percent": precision * 100,
        "Recall_Detection_Rate_percent": recall * 100,
        "F1_percent": f1 * 100,
        "False_Positive_Rate_percent": false_positive_rate * 100,
        "False_Negative_Rate_percent": false_negative_rate * 100,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Testing_Time_sec": test_time,
        "Testing_Time_ms_per_sample": (test_time * 1000.0) / len(test_df),
        "N_Test_Samples": len(test_df)
    }


def compute_forgetting_from_history(task_acc_history, current_stage):
    forgetting_values = {}

    for task_name, acc_list in task_acc_history.items():
        current_acc = acc_list[current_stage]

        if current_acc is None:
            continue

        previous_accs = [
            acc_list[i]
            for i in range(current_stage)
            if acc_list[i] is not None
        ]

        if len(previous_accs) == 0:
            continue

        best_previous = max(previous_accs)
        forgetting = max(0.0, best_previous - current_acc)
        forgetting_values[task_name] = forgetting

    if len(forgetting_values) == 0:
        average_forgetting = 0.0
    else:
        average_forgetting = float(np.mean(list(forgetting_values.values())))

    return forgetting_values, average_forgetting


# ============================================================
# 11. ONLINE DETECTION CURVES
# ============================================================

def sample_rows(df, n, rng):
    idx = rng.integers(0, len(df), size=n)
    return df.iloc[idx].copy().reset_index(drop=True)


def build_attack_stream(
    positive_df,
    negative_df,
    total_time_min=TOTAL_TIME_MIN,
    attack_start_min=ATTACK_START_MIN,
    attack_end_min=ATTACK_END_MIN,
    samples_per_min=SAMPLES_PER_MIN,
    attack_ratio=ATTACK_RATIO_DURING_ATTACK,
    seed=RANDOM_STATE
):
    rng = np.random.default_rng(seed)
    stream_parts = []

    for minute in range(total_time_min):
        if minute < attack_start_min or minute >= attack_end_min:
            block = sample_rows(negative_df, samples_per_min, rng)
            block["is_attack"] = 0
        else:
            n_attack = int(round(samples_per_min * attack_ratio))
            n_attack = min(max(n_attack, 1), samples_per_min - 1)
            n_negative = samples_per_min - n_attack

            pos_block = sample_rows(positive_df, n_attack, rng)
            pos_block["is_attack"] = 1

            neg_block = sample_rows(negative_df, n_negative, rng)
            neg_block["is_attack"] = 0

            block = pd.concat([pos_block, neg_block], axis=0).sample(
                frac=1.0,
                random_state=seed + minute
            ).reset_index(drop=True)

        block["time_min"] = minute
        stream_parts.append(block)

    return pd.concat(stream_parts, axis=0).reset_index(drop=True)


def compute_windowed_detection_curve(
    model,
    stream_df,
    positive_class_name,
    scaler,
    class_to_idx,
    total_time_min=TOTAL_TIME_MIN,
    window_size_min=WINDOW_SIZE_MIN
):
    _, pred_labels, test_time = predict_label_names(
        model,
        stream_df,
        scaler,
        class_to_idx
    )

    true_binary = (stream_df["is_attack"].values == 1).astype(int)
    pred_binary = (pred_labels == positive_class_name).astype(int)

    times = np.arange(total_time_min)
    recalls = []

    for t in times:
        start_t = max(0, t - window_size_min + 1)
        mask = (stream_df["time_min"].values >= start_t) & (stream_df["time_min"].values <= t)

        true_window = true_binary[mask]
        pred_window = pred_binary[mask]

        positives = np.sum(true_window == 1)

        if positives == 0:
            recalls.append(0.0)
        else:
            tp = np.sum((true_window == 1) & (pred_window == 1))
            fn = np.sum((true_window == 1) & (pred_window == 0))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall * 100.0)

    return pd.DataFrame({
        "time_min": times,
        "recall_percent": recalls
    }), test_time


# ============================================================
# 12. COMMUNICATION COST
# ============================================================

def full_model_comm_kb(model):
    return count_model_params(model) * FLOAT_BYTES / 1024.0


def reg_cil_comm_kb(feature_dim, current_classes):
    """
    Corrected Reg-CIL communication:
    Reg-CIL does not need to send the whole encoder.
    It sends the updated classifier head / lightweight trainable part.
    For this simple IDS, this is approximated by the classifier head.
    """
    params = classifier_head_params(feature_dim, current_classes)
    return (N_AV + 1) * params * FLOAT_BYTES / 1024.0


def full_fl_comm_kb(model):
    """
    Full FL communication:
    N AVs upload full model updates and server broadcasts aggregated model.
    """
    return (N_AV + 1) * full_model_comm_kb(model)


def dt_gcil_comm_kb(feature_dim, n_new_classes):
    """
    DT-GCIL communication:
    only trusted AVs send new-class increments, and server broadcasts increments.
    """
    trusted_avs = max(1, int(np.ceil(N_AV * TRUSTED_FRACTION)))
    params = dtgcil_increment_params(feature_dim, n_new_classes)
    return (trusted_avs + N_AV) * params * FLOAT_BYTES / 1024.0


def compute_stage_communication_table(
    model_snapshots,
    D1_test,
    D2_test,
    D3_test,
    scaler,
    class_to_idx,
    feature_dim,
    d1_class_count,
    d2_new_count,
    d3_new_count
):
    stage_names = [
        "Stage 1: Initial D1",
        "Stage 2: DoS arrival",
        "Stage 3: GAS arrival"
    ]

    stage_labels = {
        "Stage 1: Initial D1": "D1",
        "Stage 2: DoS arrival": "D1+D2",
        "Stage 3: GAS arrival": "D1+D2+D3"
    }

    stage_tests = {
        "Stage 1: Initial D1": D1_test,
        "Stage 2: DoS arrival": pd.concat([D1_test, D2_test], axis=0).reset_index(drop=True),
        "Stage 3: GAS arrival": pd.concat([D1_test, D2_test, D3_test], axis=0).reset_index(drop=True)
    }

    rows = []

    for method in METHOD_ORDER:
        cumulative_comm_kb = 0.0

        for stage_idx, stage in enumerate(stage_names):
            model = model_snapshots[method][stage]
            acc_summary, _, _ = evaluate_multiclass(
                model,
                stage_tests[stage],
                scaler,
                class_to_idx
            )

            if stage_idx == 0:
                # initial deployment: all methods send initial model
                stage_comm_kb = full_model_comm_kb(model)

            elif stage == "Stage 2: DoS arrival":
                current_classes = d1_class_count + d2_new_count

                if method == METHOD_FULL:
                    stage_comm_kb = full_fl_comm_kb(model)
                elif method == METHOD_REG:
                    stage_comm_kb = reg_cil_comm_kb(feature_dim, current_classes)
                else:
                    stage_comm_kb = dt_gcil_comm_kb(feature_dim, d2_new_count)

            else:
                current_classes = d1_class_count + d2_new_count + d3_new_count

                if method == METHOD_FULL:
                    stage_comm_kb = full_fl_comm_kb(model)
                elif method == METHOD_REG:
                    stage_comm_kb = reg_cil_comm_kb(feature_dim, current_classes)
                else:
                    stage_comm_kb = dt_gcil_comm_kb(feature_dim, d3_new_count)

            cumulative_comm_kb += stage_comm_kb

            rows.append({
                "Method": method,
                "Stage": stage,
                "Stage_Label": stage_labels[stage],
                "Accuracy_percent": acc_summary["accuracy_percent"],
                "Stage_Communication_KB": stage_comm_kb,
                "Cumulative_Communication_KB": cumulative_comm_kb
            })

    return pd.DataFrame(rows)


def compute_trust_comm_kb(feature_dim, d2_new_count, d3_new_count, normal_model_stage2, normal_model_stage3):
    """
    Trust-aware aggregation sends only selected increments.
    Normal aggregation sends full model updates from all AVs.
    """
    trust_incremental = (
        dt_gcil_comm_kb(feature_dim, d2_new_count)
        + dt_gcil_comm_kb(feature_dim, d3_new_count)
    )

    normal_full = (
        full_fl_comm_kb(normal_model_stage2)
        + full_fl_comm_kb(normal_model_stage3)
    )

    return trust_incremental, normal_full


# ============================================================
# 13. PLOTTING FUNCTIONS
# ============================================================

def plot_detection_over_time(curves_dict, y_label, save_path_no_ext):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    for method in METHOD_ORDER:
        if method not in curves_dict:
            continue

        curve_df = curves_dict[method]
        peak = np.max(curve_df["recall_percent"].values)

        ax.plot(
            curve_df["time_min"],
            curve_df["recall_percent"],
            color=MODEL_COLORS[method],
            marker=MODEL_MARKERS[method],
            markevery=35,
            label=f"{method} ({peak:.0f})"
        )

    ax.axvline(ATTACK_START_MIN, linestyle="--", linewidth=1.0, color="black")
    ax.axvline(ATTACK_END_MIN, linestyle="--", linewidth=1.0, color="black")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, TOTAL_TIME_MIN - 1)
    ax.set_ylim(-2, 105)
    ax.grid(True)

    compact_legend(ax, loc="lower right", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_binary_metric_bars(metrics_df, attack_name, save_path_no_ext):
    subset = metrics_df[metrics_df["Attack"] == attack_name].copy()

    metric_cols = [
        "Accuracy_percent",
        "Precision_percent",
        "Recall_Detection_Rate_percent",
        "F1_percent",
        "False_Positive_Rate_percent",
        "False_Negative_Rate_percent"
    ]

    labels = ["Acc", "Prec", "Rec", "F1", "FPR", "FNR"]

    x = np.arange(len(metric_cols))
    width = 0.24

    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    for i, method in enumerate(METHOD_ORDER):
        row = subset[subset["Method"] == method]
        if len(row) == 0:
            continue

        values = row[metric_cols].iloc[0].values.astype(float)

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
            color=MODEL_COLORS[method],
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")

    compact_legend(ax, loc="upper right", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_training_time_breakdown(stage_timing_df, save_path_no_ext):
    stage_order = ["Stage 1: Initial D1", "Stage 2: Learn DoS", "Stage 3: Learn GAS"]
    short_stage = ["D1", "DoS", "GAS"]

    x = np.arange(len(stage_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(3.9, 2.7))

    for i, method in enumerate(METHOD_ORDER):
        vals = []

        for stage in stage_order:
            row = stage_timing_df[
                (stage_timing_df["Stage"] == stage)
                & (stage_timing_df["Method"] == method)
            ]
            vals.append(float(row["Training_Time_sec"].values[0]))

        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=method,
            color=MODEL_COLORS[method],
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short_stage)
    ax.set_ylabel("Training Time (s)")
    ax.grid(axis="y")

    compact_legend(ax, loc="upper right", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_forgetting_bar_with_labels(forgetting_df, save_path_no_ext):
    """
    Log-scale forgetting bar chart.
    Real zero values are shown using a small visual floor because log scale
    cannot display zero. Labels still show the true values.
    """

    stage_order = ["Stage 1: Initial D1", "Stage 2: DoS arrival", "Stage 3: GAS arrival"]
    short_stage = ["D1", "DoS", "GAS"]

    x = np.arange(len(stage_order))
    width = 0.24

    visual_floor = 0.01  # only for plotting zero values on log scale

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    for i, method in enumerate(METHOD_ORDER):
        true_vals = []

        for stage in stage_order:
            row = forgetting_df[
                (forgetting_df["Stage"] == stage)
                & (forgetting_df["Method"] == method)
            ]
            true_vals.append(float(row["Average_Forgetting_percent"].values[0]))

        plot_vals = [v if v > 0 else visual_floor for v in true_vals]
        xpos = x + (i - 1) * width

        ax.bar(
            xpos,
            plot_vals,
            width,
            label=method,
            color=MODEL_COLORS[method],
            edgecolor="black",
            linewidth=0.8
        )

        for xx, y_plot, y_true in zip(xpos, plot_vals, true_vals):
            label_text = f"{y_true:.2f}"

            ax.text(
                xx,
                y_plot * 1.25,
                label_text,
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                rotation=0
            )

    ax.set_yscale("log")
    ax.set_ylim(visual_floor, max(30, forgetting_df["Average_Forgetting_percent"].max() * 2))

    ax.set_xticks(x)
    ax.set_xticklabels(short_stage)
    ax.set_ylabel("Avg. Forgetting (%)")
    ax.grid(axis="y", which="both")

    compact_legend(ax, loc="upper left", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_forgetting_taskwise_final(final_forgetting_df, save_path_no_ext):
    tasks = ["D1_Forgetting_percent", "D2_Forgetting_percent"]
    task_labels = ["D1", "DoS"]

    x = np.arange(len(tasks))
    width = 0.24

    fig, ax = plt.subplots(figsize=(3.8, 2.7))

    max_val = max(
        1.0,
        final_forgetting_df[tasks].fillna(0).values.max()
    )
    label_offset = max_val * 0.035

    for i, method in enumerate(METHOD_ORDER):
        row = final_forgetting_df[final_forgetting_df["Method"] == method]
        vals = row[tasks].iloc[0].fillna(0.0).values.astype(float)
        xpos = x + (i - 1) * width

        ax.bar(
            xpos,
            vals,
            width,
            label=method,
            color=MODEL_COLORS[method],
            edgecolor="black",
            linewidth=0.8
        )

        for xx, yy in zip(xpos, vals):
            if yy < 0.05:
                ax.plot(
                    xx,
                    0.05,
                    marker=MODEL_MARKERS[method],
                    color=MODEL_COLORS[method],
                    markersize=4.0,
                    markeredgecolor="black",
                    zorder=5
                )
                ax.text(
                    xx,
                    0.12,
                    f"{yy:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    fontweight="bold",
                    rotation=90
                )
            else:
                ax.text(
                    xx,
                    yy + label_offset,
                    f"{yy:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    fontweight="bold",
                    rotation=90
                )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Forgetting (%)")
    ax.grid(axis="y")

    compact_legend(ax, loc="upper left", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_total_training_time(timing_df, save_path_no_ext):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    x = np.arange(len(METHOD_ORDER))
    vals = []

    for m in METHOD_ORDER:
        vals.append(float(timing_df.loc[
            timing_df["Method"] == m,
            "Total_Training_Time_sec"
        ].values[0]))

    ax.bar(
        x,
        vals,
        color=[MODEL_COLORS[m] for m in METHOD_ORDER],
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Full", "Reg", "DT"])
    ax.set_ylabel("Total Train. Time (s)")
    ax.grid(axis="y")

    make_axes_bold(ax)
    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_variant_score_bars(summary_df, methods, metric_cols, xlabels, save_path_no_ext):
    x = np.arange(len(metric_cols))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    for i, method in enumerate(methods):
        row = summary_df[summary_df["Method"] == method]
        if len(row) == 0:
            continue

        vals = row[metric_cols].iloc[0].values.astype(float)

        color = "red" if method == METHOD_DTG else "blue"

        label = method.replace("DT-GCIL ", "")

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")

    compact_legend(ax, loc="lower center", ncol=2)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_trust_performance_communication(trust_df, save_path_no_ext):
    methods = [METHOD_DTG, METHOD_DTG_NOTRUST]
    perf_metrics = ["Accuracy", "Macro_F1", "FPR_Score", "Forgetting_Score"]
    perf_labels = ["Acc", "F1", "1-FPR", "1-For"]

    x = np.arange(len(perf_metrics))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(4.4, 2.8))

    for i, method in enumerate(methods):
        row = trust_df[trust_df["Method"] == method]
        vals = row[perf_metrics].iloc[0].values.astype(float)

        color = "red" if method == METHOD_DTG else "blue"
        label = "Trust" if method == METHOD_DTG else "Normal"

        ax1.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=label
        )

    ax1.set_ylabel("Performance (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()

    comm_x = len(perf_metrics) + np.array([0.0, 0.38])
    comm_vals = []
    comm_colors = []

    for method in methods:
        row = trust_df[trust_df["Method"] == method]
        comm_vals.append(float(row["Communication_KB"].values[0]))
        comm_colors.append("red" if method == METHOD_DTG else "blue")

    ax2.bar(
        comm_x,
        comm_vals,
        width=0.28,
        color=comm_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.55
    )

    ax2.set_ylabel("Comm. Cost (KB)")
    ax2.tick_params(axis="y", width=1.0, length=3.5)

    xticks_all = list(x) + list(comm_x)
    xlabels_all = perf_labels + ["C-T", "C-N"]
    ax1.set_xticks(xticks_all)
    ax1.set_xticklabels(xlabels_all)

    compact_legend(ax1, loc="lower center", ncol=2)
    make_axes_bold(ax1)
    make_axes_bold(ax2)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_accuracy_comm_dual_axis(comm_acc_df, save_path_no_ext):
    """
    Requested figure:
    x-axis: D1, D1+D2, D1+D2+D3
    left y-axis: accuracy
    right y-axis: communication cost
    """
    stage_order = ["D1", "D1+D2", "D1+D2+D3"]
    x = np.arange(len(stage_order))

    fig, ax1 = plt.subplots(figsize=(3.9, 2.7))
    ax2 = ax1.twinx()

    for method in METHOD_ORDER:
        group = comm_acc_df[comm_acc_df["Method"] == method].copy()
        group["stage_x"] = group["Stage_Label"].map({s: i for i, s in enumerate(stage_order)})
        group = group.sort_values("stage_x")

        ax1.plot(
            group["stage_x"],
            group["Accuracy_percent"],
            color=MODEL_COLORS[method],
            marker=MODEL_MARKERS[method],
            linestyle="-",
            label=f"{method} Acc"
        )

        ax2.plot(
            group["stage_x"],
            group["Cumulative_Communication_KB"],
            color=MODEL_COLORS[method],
            marker=MODEL_MARKERS[method],
            linestyle="--",
            label=f"{method} Comm"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_order)
    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Cum. Comm. Cost (KB)")
    ax1.set_ylim(0, 105)

    ax1.grid(True)

    # combined legend, compact
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # ax1.legend(
    #     h1 + h2,
    #     l1 + l2,
    #     loc="lower center",
    #     ncol=2,
    #     fontsize=6.2,
    #     frameon=True,
    #     framealpha=1.0,
    #     edgecolor="black",
    #     borderpad=0.25,
    #     labelspacing=0.20,
    #     columnspacing=0.55,
    #     handlelength=1.2
    # )

    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="center left",
        bbox_to_anchor=(0.02, 0.50),
        ncol=1,
        fontsize=6.2,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        borderpad=0.25,
        labelspacing=0.20,
        columnspacing=0.55,
        handlelength=1.2
    )

    make_axes_bold(ax1)
    make_axes_bold(ax2)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def plot_stage_communication_bar(comm_acc_df, save_path_no_ext):
    stage_order = ["D1", "D1+D2", "D1+D2+D3"]

    x = np.arange(len(stage_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(3.9, 2.7))

    for i, method in enumerate(METHOD_ORDER):
        vals = []

        for stage in stage_order:
            row = comm_acc_df[
                (comm_acc_df["Stage_Label"] == stage)
                & (comm_acc_df["Method"] == method)
            ]
            vals.append(float(row["Stage_Communication_KB"].values[0]))

        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=method,
            color=MODEL_COLORS[method],
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel("Stage Comm. Cost (KB)")
    ax.grid(axis="y")

    compact_legend(ax, loc="upper left", ncol=1)
    make_axes_bold(ax)

    fig.tight_layout()
    save_figure_both(fig, save_path_no_ext)
    plt.close(fig)


def save_impact_tables_as_latex(gan_table, trust_table, output_dir):
    gan_table.to_latex(
        os.path.join(output_dir, "gan_impact_table.tex"),
        index=False,
        float_format="%.3f"
    )

    trust_table.to_latex(
        os.path.join(output_dir, "trust_impact_table.tex"),
        index=False,
        float_format="%.3f"
    )


# ============================================================
# 14. MAIN
# ============================================================

print(f"Using device: {DEVICE}")

D1 = load_dataset(D1_PATH, MAX_ROWS_PER_DATASET)
D2 = load_dataset(D2_PATH, MAX_ROWS_PER_DATASET)
D3 = load_dataset(D3_PATH, MAX_ROWS_PER_DATASET)

print("\nD1 classes:")
print(D1["target"].value_counts())

print("\nD2 classes:")
print(D2["target"].value_counts())

print("\nD3 classes:")
print(D3["target"].value_counts())

D1_train, D1_test = safe_split(D1, TEST_SIZE_D1)
D2_adapt, D2_test = safe_split(D2, TEST_SIZE_NEW)
D3_adapt, D3_test = safe_split(D3, TEST_SIZE_NEW)

class_order = build_class_order(D1, D2, D3)
class_to_idx = {c: i for i, c in enumerate(class_order)}

d1_classes = sorted(D1["target"].unique().tolist())
d2_new_classes = [c for c in sorted(D2["target"].unique().tolist()) if c not in d1_classes]
d3_new_classes = [c for c in sorted(D3["target"].unique().tolist()) if c not in d1_classes + d2_new_classes]

dos_class = D2["target"].iloc[0]
gas_class = D3["target"].iloc[0]

print("\nClass order:")
for i, c in enumerate(class_order):
    print(i, c)

print(f"\nDoS class: {dos_class}")
print(f"GAS class: {gas_class}")

scaler = StandardScaler()
scaler.fit(D1_train[FEATURE_COLS])

replay_main = make_replay_buffer(D1_train, REPLAY_PER_CLASS)
replay_nogan = make_replay_buffer(D1_train, REPLAY_PER_CLASS)
replay_notrust = make_replay_buffer(D1_train, REPLAY_PER_CLASS)

stage_timing_rows = []


# ============================================================
# 15. STAGE 1: INITIAL D1 TRAINING
# ============================================================

print("\nStage 1: Initial training on D1...")

dtgcil_model, dtgcil_base_time = train_model_from_scratch(
    D1_train,
    scaler,
    class_to_idx,
    num_classes=len(d1_classes),
    epochs=BASE_EPOCHS
)

regcil_model, regcil_base_time = train_model_from_scratch(
    D1_train,
    scaler,
    class_to_idx,
    num_classes=len(d1_classes),
    epochs=BASE_EPOCHS
)

fullfl_model_stage0, fullfl_time_stage0 = train_model_from_scratch(
    D1_train,
    scaler,
    class_to_idx,
    num_classes=len(d1_classes),
    epochs=FULL_FL_EPOCHS
)

dtgcil_nogan_model = copy.deepcopy(dtgcil_model).to(DEVICE)
dtgcil_notrust_model = copy.deepcopy(dtgcil_model).to(DEVICE)

stage_timing_rows.extend([
    {"Stage": "Stage 1: Initial D1", "Method": METHOD_DTG, "Training_Time_sec": dtgcil_base_time},
    {"Stage": "Stage 1: Initial D1", "Method": METHOD_REG, "Training_Time_sec": regcil_base_time},
    {"Stage": "Stage 1: Initial D1", "Method": METHOD_FULL, "Training_Time_sec": fullfl_time_stage0},
])

model_snapshots = {
    METHOD_FULL: {"Stage 1: Initial D1": copy.deepcopy(fullfl_model_stage0).to(DEVICE)},
    METHOD_REG: {"Stage 1: Initial D1": copy.deepcopy(regcil_model).to(DEVICE)},
    METHOD_DTG: {"Stage 1: Initial D1": copy.deepcopy(dtgcil_model).to(DEVICE)},
    METHOD_DTG_NOGAN: {"Stage 1: Initial D1": copy.deepcopy(dtgcil_nogan_model).to(DEVICE)},
    METHOD_DTG_NOTRUST: {"Stage 1: Initial D1": copy.deepcopy(dtgcil_notrust_model).to(DEVICE)}
}


# ============================================================
# 16. STAGE 2: DoS ARRIVES
# ============================================================

print("\nStage 2: DoS arrives...")

fullfl_train_stage1 = pd.concat([D1_train, D2_adapt], axis=0).reset_index(drop=True)

fullfl_model_stage1, fullfl_time_stage1 = train_model_from_scratch(
    fullfl_train_stage1,
    scaler,
    class_to_idx,
    num_classes=len(d1_classes) + len(d2_new_classes),
    epochs=FULL_FL_EPOCHS
)

regcil_model.expand_classes(len(d2_new_classes))
regcil_time_stage1 = update_regularization_cil(
    regcil_model,
    D2_adapt,
    scaler,
    class_to_idx
)

dtgcil_model.expand_classes(len(d2_new_classes))
dtgcil_time_stage1 = update_dtgcil(
    dtgcil_model,
    D2_adapt,
    replay_main,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes),
    use_gan=True,
    trust_aware=True
)

dtgcil_nogan_model.expand_classes(len(d2_new_classes))
dtgcil_nogan_time_stage1 = update_dtgcil(
    dtgcil_nogan_model,
    D2_adapt,
    replay_nogan,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes),
    use_gan=False,
    trust_aware=True
)

dtgcil_notrust_model.expand_classes(len(d2_new_classes))
dtgcil_notrust_time_stage1 = update_dtgcil(
    dtgcil_notrust_model,
    D2_adapt,
    replay_notrust,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes),
    use_gan=True,
    trust_aware=False
)

replay_main = make_replay_buffer(
    pd.concat([replay_main, D2_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

replay_nogan = make_replay_buffer(
    pd.concat([replay_nogan, D2_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

replay_notrust = make_replay_buffer(
    pd.concat([replay_notrust, D2_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

stage_timing_rows.extend([
    {"Stage": "Stage 2: Learn DoS", "Method": METHOD_DTG, "Training_Time_sec": dtgcil_time_stage1},
    {"Stage": "Stage 2: Learn DoS", "Method": METHOD_REG, "Training_Time_sec": regcil_time_stage1},
    {"Stage": "Stage 2: Learn DoS", "Method": METHOD_FULL, "Training_Time_sec": fullfl_time_stage1},
])

model_snapshots[METHOD_FULL]["Stage 2: DoS arrival"] = copy.deepcopy(fullfl_model_stage1).to(DEVICE)
model_snapshots[METHOD_REG]["Stage 2: DoS arrival"] = copy.deepcopy(regcil_model).to(DEVICE)
model_snapshots[METHOD_DTG]["Stage 2: DoS arrival"] = copy.deepcopy(dtgcil_model).to(DEVICE)
model_snapshots[METHOD_DTG_NOGAN]["Stage 2: DoS arrival"] = copy.deepcopy(dtgcil_nogan_model).to(DEVICE)
model_snapshots[METHOD_DTG_NOTRUST]["Stage 2: DoS arrival"] = copy.deepcopy(dtgcil_notrust_model).to(DEVICE)


# ============================================================
# 17. DoS METRICS
# ============================================================

dos_negative_df = D1_test.copy()

dos_metrics_all = {
    METHOD_FULL: binary_attack_metrics(fullfl_model_stage1, D2_test, dos_negative_df, dos_class, scaler, class_to_idx),
    METHOD_REG: binary_attack_metrics(regcil_model, D2_test, dos_negative_df, dos_class, scaler, class_to_idx),
    METHOD_DTG: binary_attack_metrics(dtgcil_model, D2_test, dos_negative_df, dos_class, scaler, class_to_idx),
    METHOD_DTG_NOGAN: binary_attack_metrics(dtgcil_nogan_model, D2_test, dos_negative_df, dos_class, scaler, class_to_idx),
    METHOD_DTG_NOTRUST: binary_attack_metrics(dtgcil_notrust_model, D2_test, dos_negative_df, dos_class, scaler, class_to_idx)
}

dos_stream = build_attack_stream(
    positive_df=D2_test,
    negative_df=dos_negative_df,
    seed=RANDOM_STATE
)

dos_curves = {}
dos_stream_test_times = {}

for method_name, model in [
    (METHOD_FULL, fullfl_model_stage1),
    (METHOD_REG, regcil_model),
    (METHOD_DTG, dtgcil_model)
]:
    curve_df, t_stream = compute_windowed_detection_curve(
        model,
        dos_stream,
        dos_class,
        scaler,
        class_to_idx
    )
    dos_curves[method_name] = curve_df
    dos_stream_test_times[method_name] = t_stream


# ============================================================
# 18. STAGE 3: GAS ARRIVES
# ============================================================

print("\nStage 3: GAS spoofing arrives...")

fullfl_train_stage2 = pd.concat([D1_train, D2_adapt, D3_adapt], axis=0).reset_index(drop=True)

fullfl_model_stage2, fullfl_time_stage2 = train_model_from_scratch(
    fullfl_train_stage2,
    scaler,
    class_to_idx,
    num_classes=len(d1_classes) + len(d2_new_classes) + len(d3_new_classes),
    epochs=FULL_FL_EPOCHS
)

regcil_model.expand_classes(len(d3_new_classes))
regcil_time_stage2 = update_regularization_cil(
    regcil_model,
    D3_adapt,
    scaler,
    class_to_idx
)

dtgcil_model.expand_classes(len(d3_new_classes))
dtgcil_time_stage2 = update_dtgcil(
    dtgcil_model,
    D3_adapt,
    replay_main,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes) + len(d2_new_classes),
    use_gan=True,
    trust_aware=True
)

dtgcil_nogan_model.expand_classes(len(d3_new_classes))
dtgcil_nogan_time_stage2 = update_dtgcil(
    dtgcil_nogan_model,
    D3_adapt,
    replay_nogan,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes) + len(d2_new_classes),
    use_gan=False,
    trust_aware=True
)

dtgcil_notrust_model.expand_classes(len(d3_new_classes))
dtgcil_notrust_time_stage2 = update_dtgcil(
    dtgcil_notrust_model,
    D3_adapt,
    replay_notrust,
    scaler,
    class_to_idx,
    old_class_count=len(d1_classes) + len(d2_new_classes),
    use_gan=True,
    trust_aware=False
)

replay_main = make_replay_buffer(
    pd.concat([replay_main, D3_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

replay_nogan = make_replay_buffer(
    pd.concat([replay_nogan, D3_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

replay_notrust = make_replay_buffer(
    pd.concat([replay_notrust, D3_adapt], axis=0).reset_index(drop=True),
    REPLAY_PER_CLASS
)

stage_timing_rows.extend([
    {"Stage": "Stage 3: Learn GAS", "Method": METHOD_DTG, "Training_Time_sec": dtgcil_time_stage2},
    {"Stage": "Stage 3: Learn GAS", "Method": METHOD_REG, "Training_Time_sec": regcil_time_stage2},
    {"Stage": "Stage 3: Learn GAS", "Method": METHOD_FULL, "Training_Time_sec": fullfl_time_stage2},
])

model_snapshots[METHOD_FULL]["Stage 3: GAS arrival"] = copy.deepcopy(fullfl_model_stage2).to(DEVICE)
model_snapshots[METHOD_REG]["Stage 3: GAS arrival"] = copy.deepcopy(regcil_model).to(DEVICE)
model_snapshots[METHOD_DTG]["Stage 3: GAS arrival"] = copy.deepcopy(dtgcil_model).to(DEVICE)
model_snapshots[METHOD_DTG_NOGAN]["Stage 3: GAS arrival"] = copy.deepcopy(dtgcil_nogan_model).to(DEVICE)
model_snapshots[METHOD_DTG_NOTRUST]["Stage 3: GAS arrival"] = copy.deepcopy(dtgcil_notrust_model).to(DEVICE)


# ============================================================
# 19. GAS METRICS
# ============================================================

gas_negative_df = pd.concat([D1_test, D2_test], axis=0).reset_index(drop=True)

gas_metrics_all = {
    METHOD_FULL: binary_attack_metrics(fullfl_model_stage2, D3_test, gas_negative_df, gas_class, scaler, class_to_idx),
    METHOD_REG: binary_attack_metrics(regcil_model, D3_test, gas_negative_df, gas_class, scaler, class_to_idx),
    METHOD_DTG: binary_attack_metrics(dtgcil_model, D3_test, gas_negative_df, gas_class, scaler, class_to_idx),
    METHOD_DTG_NOGAN: binary_attack_metrics(dtgcil_nogan_model, D3_test, gas_negative_df, gas_class, scaler, class_to_idx),
    METHOD_DTG_NOTRUST: binary_attack_metrics(dtgcil_notrust_model, D3_test, gas_negative_df, gas_class, scaler, class_to_idx)
}

gas_stream = build_attack_stream(
    positive_df=D3_test,
    negative_df=gas_negative_df,
    seed=RANDOM_STATE + 1
)

gas_curves = {}
gas_stream_test_times = {}

for method_name, model in [
    (METHOD_FULL, fullfl_model_stage2),
    (METHOD_REG, regcil_model),
    (METHOD_DTG, dtgcil_model)
]:
    curve_df, t_stream = compute_windowed_detection_curve(
        model,
        gas_stream,
        gas_class,
        scaler,
        class_to_idx
    )
    gas_curves[method_name] = curve_df
    gas_stream_test_times[method_name] = t_stream


# ============================================================
# 20. FINAL MULTICLASS TEST
# ============================================================

final_test = pd.concat([D1_test, D2_test, D3_test], axis=0).reset_index(drop=True)

models_final = {
    METHOD_FULL: fullfl_model_stage2,
    METHOD_REG: regcil_model,
    METHOD_DTG: dtgcil_model,
    METHOD_DTG_NOGAN: dtgcil_nogan_model,
    METHOD_DTG_NOTRUST: dtgcil_notrust_model
}

final_summary_rows = {}
classification_reports = {}

for method, model in models_final.items():
    summary, true_labels, pred_labels = evaluate_multiclass(
        model,
        final_test,
        scaler,
        class_to_idx
    )

    final_summary_rows[method] = {
        "Method": method,
        "Final_Multiclass_Accuracy_percent": summary["accuracy_percent"],
        "Final_Macro_F1_percent": summary["macro_f1_percent"],
        "Final_Testing_Time_sec": summary["testing_time_sec"],
        "Final_Testing_Time_ms_per_sample": summary["testing_time_ms_per_sample"],
        "N_Final_Test_Samples": summary["n_samples"]
    }

    classification_reports[method] = classification_report(
        true_labels,
        pred_labels,
        zero_division=0
    )

final_multiclass_summary = pd.DataFrame(list(final_summary_rows.values()))


# ============================================================
# 21. MIXED BINARY METRICS
# ============================================================

rows = []

for method, values in dos_metrics_all.items():
    row = {"Attack": "DoS", "Method": method}
    row.update(values)
    rows.append(row)

for method, values in gas_metrics_all.items():
    row = {"Attack": "GAS Spoofing", "Method": method}
    row.update(values)
    rows.append(row)

mixed_binary_metrics_df = pd.DataFrame(rows)

mixed_binary_metrics_main_df = mixed_binary_metrics_df[
    mixed_binary_metrics_df["Method"].isin(METHOD_ORDER)
].copy()


# ============================================================
# 22. TIMING SUMMARY
# ============================================================

stage_timing_df = pd.DataFrame(stage_timing_rows)

training_totals = stage_timing_df.groupby("Method", as_index=False)["Training_Time_sec"].sum()
training_totals = training_totals.rename(columns={"Training_Time_sec": "Total_Training_Time_sec"})

timing_summary_df = training_totals.merge(
    final_multiclass_summary[[
        "Method",
        "Final_Testing_Time_sec",
        "Final_Testing_Time_ms_per_sample"
    ]],
    on="Method",
    how="left"
)

timing_summary_df["DoS_Stream_Test_Time_sec"] = timing_summary_df["Method"].map(dos_stream_test_times)
timing_summary_df["GAS_Stream_Test_Time_sec"] = timing_summary_df["Method"].map(gas_stream_test_times)


# ============================================================
# 23. FORGETTING ANALYSIS
# ============================================================

print("\nComputing forgetting analysis...")

stages_for_forgetting = [
    "Stage 1: Initial D1",
    "Stage 2: DoS arrival",
    "Stage 3: GAS arrival"
]

tasks_by_stage = {
    "D1 old classes": D1_test,
    "D2 DoS": D2_test,
    "D3 GAS": D3_test
}

learned_tasks_at_stage = {
    "Stage 1: Initial D1": ["D1 old classes"],
    "Stage 2: DoS arrival": ["D1 old classes", "D2 DoS"],
    "Stage 3: GAS arrival": ["D1 old classes", "D2 DoS", "D3 GAS"]
}

task_accuracy_rows = []
forgetting_rows = []

for method_name, stage_models in model_snapshots.items():

    task_acc_history = {
        "D1 old classes": [],
        "D2 DoS": [],
        "D3 GAS": []
    }

    for stage_idx, stage_name in enumerate(stages_for_forgetting):

        model_at_stage = stage_models[stage_name]
        learned_tasks = learned_tasks_at_stage[stage_name]

        for task_name in ["D1 old classes", "D2 DoS", "D3 GAS"]:

            if task_name in learned_tasks:
                task_df = tasks_by_stage[task_name]

                result = evaluate_task_accuracy(
                    model_at_stage,
                    task_df,
                    scaler,
                    class_to_idx
                )

                acc = result["accuracy_percent"]
                f1 = result["macro_f1_percent"]

                task_acc_history[task_name].append(acc)

                task_accuracy_rows.append({
                    "Method": method_name,
                    "Stage": stage_name,
                    "Task": task_name,
                    "Accuracy_percent": acc,
                    "Macro_F1_percent": f1
                })

            else:
                task_acc_history[task_name].append(None)

                task_accuracy_rows.append({
                    "Method": method_name,
                    "Stage": stage_name,
                    "Task": task_name,
                    "Accuracy_percent": np.nan,
                    "Macro_F1_percent": np.nan
                })

        forgetting_values, avg_forgetting = compute_forgetting_from_history(
            task_acc_history,
            current_stage=stage_idx
        )

        forgetting_rows.append({
            "Method": method_name,
            "Stage": stage_name,
            "Average_Forgetting_percent": avg_forgetting,
            "D1_Forgetting_percent": forgetting_values.get("D1 old classes", np.nan),
            "D2_Forgetting_percent": forgetting_values.get("D2 DoS", np.nan),
            "D3_Forgetting_percent": forgetting_values.get("D3 GAS", np.nan)
        })

task_accuracy_df = pd.DataFrame(task_accuracy_rows)
forgetting_df = pd.DataFrame(forgetting_rows)

forgetting_main_df = forgetting_df[
    forgetting_df["Method"].isin(METHOD_ORDER)
].copy()

final_forgetting_df = forgetting_df[
    forgetting_df["Stage"] == "Stage 3: GAS arrival"
].copy()

final_forgetting_df = final_forgetting_df.rename(
    columns={"Average_Forgetting_percent": "Final_Average_Forgetting_percent"}
)

final_forgetting_main_df = final_forgetting_df[
    final_forgetting_df["Method"].isin(METHOD_ORDER)
].copy()


# ============================================================
# 24. GAN, TRUST, AND COMMUNICATION TABLES
# ============================================================

def average_fpr_for_method(method_name):
    rows_m = mixed_binary_metrics_df[mixed_binary_metrics_df["Method"] == method_name]
    return float(rows_m["False_Positive_Rate_percent"].mean())


def average_f1_for_method(method_name):
    rows_m = mixed_binary_metrics_df[mixed_binary_metrics_df["Method"] == method_name]
    return float(rows_m["F1_percent"].mean())


def final_forgetting_for_method(method_name):
    row = final_forgetting_df[final_forgetting_df["Method"] == method_name]
    return float(row["Final_Average_Forgetting_percent"].values[0])


feature_dim = dtgcil_model.classifier.in_features
d1_count = len(d1_classes)
d2_count = len(d2_new_classes)
d3_count = len(d3_new_classes)

comm_acc_df = compute_stage_communication_table(
    model_snapshots=model_snapshots,
    D1_test=D1_test,
    D2_test=D2_test,
    D3_test=D3_test,
    scaler=scaler,
    class_to_idx=class_to_idx,
    feature_dim=feature_dim,
    d1_class_count=d1_count,
    d2_new_count=d2_count,
    d3_new_count=d3_count
)

trust_comm_kb, normal_comm_kb = compute_trust_comm_kb(
    feature_dim=feature_dim,
    d2_new_count=d2_count,
    d3_new_count=d3_count,
    normal_model_stage2=dtgcil_notrust_model,
    normal_model_stage3=dtgcil_notrust_model
)

variant_summary_rows = []

for method in [METHOD_DTG, METHOD_DTG_NOGAN, METHOD_DTG_NOTRUST]:
    final_row = final_multiclass_summary[
        final_multiclass_summary["Method"] == method
    ].iloc[0]

    avg_fpr = average_fpr_for_method(method)
    avg_f1 = average_f1_for_method(method)
    avg_forgetting = final_forgetting_for_method(method)

    if method == METHOD_DTG:
        comm_kb = trust_comm_kb
    elif method == METHOD_DTG_NOTRUST:
        comm_kb = normal_comm_kb
    else:
        comm_kb = trust_comm_kb

    variant_summary_rows.append({
        "Method": method,
        "Final_Accuracy_percent": final_row["Final_Multiclass_Accuracy_percent"],
        "Final_Macro_F1_percent": final_row["Final_Macro_F1_percent"],
        "Avg_Binary_F1_percent": avg_f1,
        "Avg_FPR_percent": avg_fpr,
        "Final_Avg_Forgetting_percent": avg_forgetting,
        "Communication_KB": comm_kb,
        "FPR_Score": max(0.0, 100.0 - avg_fpr),
        "Forgetting_Score": max(0.0, 100.0 - avg_forgetting)
    })

variant_summary_df = pd.DataFrame(variant_summary_rows)

gan_impact_table = variant_summary_df[
    variant_summary_df["Method"].isin([METHOD_DTG, METHOD_DTG_NOGAN])
][[
    "Method",
    "Final_Accuracy_percent",
    "Final_Macro_F1_percent",
    "Avg_Binary_F1_percent",
    "Avg_FPR_percent",
    "Final_Avg_Forgetting_percent",
    "Communication_KB"
]].copy()

trust_impact_table = variant_summary_df[
    variant_summary_df["Method"].isin([METHOD_DTG, METHOD_DTG_NOTRUST])
][[
    "Method",
    "Final_Accuracy_percent",
    "Final_Macro_F1_percent",
    "Avg_Binary_F1_percent",
    "Avg_FPR_percent",
    "Final_Avg_Forgetting_percent",
    "Communication_KB"
]].copy()


# ============================================================
# 25. SAVE CSV + LATEX TABLES
# ============================================================

mixed_binary_metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "mixed_binary_ids_metrics_all.csv"),
    index=False
)

mixed_binary_metrics_main_df.to_csv(
    os.path.join(OUTPUT_DIR, "mixed_binary_ids_metrics_main.csv"),
    index=False
)

final_multiclass_summary.to_csv(
    os.path.join(OUTPUT_DIR, "final_multiclass_summary.csv"),
    index=False
)

stage_timing_df.to_csv(
    os.path.join(OUTPUT_DIR, "stage_training_times.csv"),
    index=False
)

timing_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "timing_summary.csv"),
    index=False
)

task_accuracy_df.to_csv(
    os.path.join(OUTPUT_DIR, "task_accuracy_history.csv"),
    index=False
)

forgetting_df.to_csv(
    os.path.join(OUTPUT_DIR, "forgetting_analysis.csv"),
    index=False
)

final_forgetting_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_forgetting_summary.csv"),
    index=False
)

variant_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "gan_trust_variant_summary.csv"),
    index=False
)

gan_impact_table.to_csv(
    os.path.join(OUTPUT_DIR, "gan_impact_table.csv"),
    index=False
)

trust_impact_table.to_csv(
    os.path.join(OUTPUT_DIR, "trust_impact_table.csv"),
    index=False
)

comm_acc_df.to_csv(
    os.path.join(OUTPUT_DIR, "accuracy_vs_communication.csv"),
    index=False
)

save_impact_tables_as_latex(
    gan_impact_table,
    trust_impact_table,
    OUTPUT_DIR
)

for method, curve_df in dos_curves.items():
    fname = f"dos_curve_{method.replace(' ', '_').replace('-', '_')}.csv"
    curve_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

for method, curve_df in gas_curves.items():
    fname = f"gas_curve_{method.replace(' ', '_').replace('-', '_')}.csv"
    curve_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)


# ============================================================
# 26. PRINT RESULTS
# ============================================================

print("\nMixed binary IDS metrics:")
print(mixed_binary_metrics_df)

print("\nFinal multiclass summary:")
print(final_multiclass_summary)

print("\nTraining and testing time summary:")
print(timing_summary_df)

print("\nForgetting analysis:")
print(forgetting_df)

print("\nAccuracy vs communication:")
print(comm_acc_df)

print("\nGAN impact table:")
print(gan_impact_table)

print("\nTrust-aware aggregation impact table:")
print(trust_impact_table)

for method, report in classification_reports.items():
    print(f"\nFinal classification report: {method}")
    print(report)


# ============================================================
# 27. FIGURES
# ============================================================

plot_detection_over_time(
    dos_curves,
    y_label="Recall (%)",
    save_path_no_ext=os.path.join(OUTPUT_DIR, "fig1_dos_detection_over_time")
)

plot_detection_over_time(
    gas_curves,
    y_label="Recall (%)",
    save_path_no_ext=os.path.join(OUTPUT_DIR, "fig2_gas_detection_over_time")
)

plot_binary_metric_bars(
    mixed_binary_metrics_main_df,
    "DoS",
    os.path.join(OUTPUT_DIR, "fig3_dos_binary_metrics")
)

plot_binary_metric_bars(
    mixed_binary_metrics_main_df,
    "GAS Spoofing",
    os.path.join(OUTPUT_DIR, "fig4_gas_binary_metrics")
)

plot_total_training_time(
    timing_summary_df,
    os.path.join(OUTPUT_DIR, "fig5_total_training_time")
)

plot_training_time_breakdown(
    stage_timing_df,
    os.path.join(OUTPUT_DIR, "fig7_training_time_breakdown")
)

plot_forgetting_bar_with_labels(
    forgetting_main_df,
    os.path.join(OUTPUT_DIR, "fig8_forgetting_bar_with_labels")
)

plot_forgetting_taskwise_final(
    final_forgetting_main_df,
    os.path.join(OUTPUT_DIR, "fig9_taskwise_final_forgetting")
)

plot_variant_score_bars(
    gan_impact_table.rename(columns={
        "Final_Accuracy_percent": "Accuracy",
        "Final_Macro_F1_percent": "Macro_F1",
        "Avg_FPR_percent": "Avg_FPR",
        "Final_Avg_Forgetting_percent": "Average_Forgetting"
    }).assign(
        FPR_Score=lambda x: 100.0 - x["Avg_FPR"],
        Forgetting_Score=lambda x: 100.0 - x["Average_Forgetting"]
    ),
    methods=[METHOD_DTG, METHOD_DTG_NOGAN],
    metric_cols=["Accuracy", "Macro_F1", "FPR_Score", "Forgetting_Score"],
    xlabels=["Acc", "F1", "1-FPR", "1-For"],
    save_path_no_ext=os.path.join(OUTPUT_DIR, "fig10_gan_impact")
)

plot_variant_score_bars(
    trust_impact_table.rename(columns={
        "Final_Accuracy_percent": "Accuracy",
        "Final_Macro_F1_percent": "Macro_F1",
        "Avg_FPR_percent": "Avg_FPR",
        "Final_Avg_Forgetting_percent": "Average_Forgetting"
    }).assign(
        FPR_Score=lambda x: 100.0 - x["Avg_FPR"],
        Forgetting_Score=lambda x: 100.0 - x["Average_Forgetting"]
    ),
    methods=[METHOD_DTG, METHOD_DTG_NOTRUST],
    metric_cols=["Accuracy", "Macro_F1", "FPR_Score", "Forgetting_Score"],
    xlabels=["Acc", "F1", "1-FPR", "1-For"],
    save_path_no_ext=os.path.join(OUTPUT_DIR, "fig11_trust_performance_only")
)

plot_trust_performance_communication(
    trust_impact_table.rename(columns={
        "Final_Accuracy_percent": "Accuracy",
        "Final_Macro_F1_percent": "Macro_F1",
        "Avg_FPR_percent": "Avg_FPR",
        "Final_Avg_Forgetting_percent": "Average_Forgetting"
    }).assign(
        FPR_Score=lambda x: 100.0 - x["Avg_FPR"],
        Forgetting_Score=lambda x: 100.0 - x["Average_Forgetting"]
    ),
    os.path.join(OUTPUT_DIR, "fig12_trust_performance_communication")
)

plot_accuracy_comm_dual_axis(
    comm_acc_df,
    os.path.join(OUTPUT_DIR, "fig13_accuracy_communication_dual_axis")
)

plot_stage_communication_bar(
    comm_acc_df,
    os.path.join(OUTPUT_DIR, "fig14_stage_communication_cost")
)


# ============================================================
# 28. DONE
# ============================================================

print("\nSaved files in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", f)