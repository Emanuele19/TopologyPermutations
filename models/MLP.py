import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.optim as optim
from .Model import SimpleModelInterface

class MLP(SimpleModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)  # logits per BCEWithLogitsLoss

    def train_model(self, data, optimizer, criterion, epochs=200, patience=10, log=False):
        self.train()
        loss_values, val_loss_values = [], []
        best_val = float('inf'); patience_ctr = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.forward(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss.backward(); optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self.forward(data.x[data.val_mask])
                val_loss = criterion(val_logits, data.y[data.val_mask].float())
            self.train()

            loss_values.append(loss.item()); val_loss_values.append(val_loss.item())
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss; patience_ctr = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if self.log: print(f"Early stopping @ epoch {epoch}")
                    break

        # ripristina best
        self.load_state_dict(best_state)

        if log:
            plt.figure()
            plt.plot(loss_values, label='Train'); plt.plot(val_loss_values, label='Val', linestyle='dashed')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('MLP Loss'); plt.legend(); plt.show()

    def k_fold_cross_validation(self, data, k=5, epochs=100, learning_rate=0.001, patience=10, log=False):
        # k-fold solo sul "train_data_tab" (già senza common)
        y_np = data.y.cpu().numpy().astype(int)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        best_model = None; best_f1 = -1.0
        acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
        best_fold_metrics = {}; best_cm = None

        for fold, (tr_idx, va_idx) in enumerate(kf.split(data.x, y_np), start=1):
            if log: print(f"\n🔹 Fold {fold}/{k}")

            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask   = torch.zeros_like(train_mask)
            train_mask[torch.as_tensor(tr_idx)] = True
            val_mask[torch.as_tensor(va_idx)]   = True
            data.train_mask = train_mask
            data.val_mask   = val_mask

            model = MLP(self.in_channels, self.hidden_channels, self.out_channels, self.dropout)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            criterion = torch.nn.BCEWithLogitsLoss()

            model.train_model(data, optimizer, criterion, epochs, patience)

            model.eval()
            with torch.no_grad():
                logits = model(data.x[data.val_mask])
                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > 0.5).astype(int)
                y_true = data.y[data.val_mask].cpu().numpy().astype(int)

                acc = accuracy_score(y_true, pred)
                pre = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                f1  = f1_score(y_true, pred, zero_division=0)
                cm  = confusion_matrix(y_true, pred, labels=[0,1])

            acc_scores.append(acc); prec_scores.append(pre); rec_scores.append(rec); f1_scores.append(f1)
            if log:
                print(f"Acc={acc:.4f} Prec={pre:.4f} Rec={rec:.4f} F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1; best_model = model
                best_fold_metrics = {"Accuracy": acc, "Precision": pre, "Recall": rec, "F1-Score": f1}
                best_cm = cm

        if log:
            print("\n📊 K-Fold (media ± std)")
            print(f"Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
            print(f"Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
            print(f"Recall   : {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
            print(f"F1-Score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

            if best_cm is not None:
                plt.figure(figsize=(4,4))
                sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
                plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Best Fold Confusion Matrix"); plt.show()

        return best_model, best_fold_metrics
