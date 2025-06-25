import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from Model import ModelInterface

class GCN_Sage(ModelInterface):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0):
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.sigmoid(x)      
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x.view(-1)  # Output logits
    
    def train_model(self, data, optimizer, criterion, epochs=200, patience=10):
        self.train()
        loss_values = []
        val_loss_values = []
        best_val_loss = float('inf')
        patience_counter = 0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        for epoch in range(epochs):
            optimizer.zero_grad()

            train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
            edge_index_sub, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=True)
            x_sub = data.x[train_nodes]
            y_sub = data.y[train_nodes]

            out = self.forward(x_sub, edge_index_sub)
            train_loss = criterion(out, y_sub.float())
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_out = self.forward(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask].float())
            self.train()

            loss_values.append(train_loss.item())
            val_loss_values.append(val_loss.item())

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and self.log:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if epoch % 10 == 0 and self.log:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if self.log:
            plt.figure()
            plt.plot(range(len(loss_values)), loss_values, label='Training Loss')
            plt.plot(range(len(val_loss_values)), val_loss_values, label='Validation Loss', linestyle='dashed')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training & Validation Loss Curve')
            plt.legend()
            plt.show()
    
    def k_fold_cross_validation(self, data, k=5, epochs=50, learning_rate=0.001, patience=10):
        """
        Questa funzione imposta un k-fold cross validation stratificato assicurandosi di avere le proteine
        in comune sempre nel test set (sarebbe un data leakage altrimenti).
        Esegue i k training e stampa le metriche di valutazione, la media di queste ed il miglior modello allenato.
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        indices = np.arange(data.x.shape[0])
        common_proteins_mask = np.array(["_AD" in name or "_PD" in name for name in data.name])
        best_model = None
        best_f1 = 0
        acc_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
        best_fold_metrics = {}
        best_cm = None

        # Fai un k-fold sul training dataset escludendo le proteine in comune
        usable_mask = (~common_proteins_mask) & data.kfold_usable_mask.cpu().numpy()
        usable_indices = np.where(usable_mask)[0]
        for fold, (train_idx, val_idx) in enumerate(kf.split(data.x[usable_mask], data.y[usable_mask].cpu().numpy())):   
            # Mappa indici locali a indici globali
            # N.B. Quando faccio qualcosa del tipo data.x[mask] sto selezionando un nuovo sottoinsieme del grafo dove ad ogni nodo
            #   verrà assegnato un nuovo indice "locale" perdendo il riferimento agli indici prima della selezione detti "globali".
            train_idx_global = usable_indices[train_idx]
            val_idx_global = usable_indices[val_idx]         
            if self.log: 
                print(f"\n🔹 Fold {fold+1}/{k}")
            train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            
            # Aggiungi le proteine comuni al training set
            train_mask[common_proteins_mask] = True
            train_mask[train_idx_global] = True
            val_mask[val_idx_global] = True
            
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = ~train_mask

            model = GCN_Sage(in_channels=self.conv1.in_channels, 
                        hidden_channels=self.conv2.out_channels, 
                        out_channels=self.conv3.out_channels, 
                        dropout=self.dropout)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            criterion = torch.nn.BCEWithLogitsLoss()
            model.train_model(data, optimizer, criterion, epochs, patience)

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                out = torch.sigmoid(out)
                pred = (out[val_mask] > 0.5).int()
                y_true = data.y[val_mask].cpu().numpy()
                y_pred = pred.cpu().numpy()

                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                acc_scores.append(acc)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                if self.log:
                    print(f"Accuracy: {acc:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1-Score: {f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_fold_metrics = {
                        "Accuracy": acc,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1,
                        "training_fold": train_idx,
                        "val_fold": val_idx
                    }
                    best_cm = cm

        if self.log:
            print("\n📊 **Statistiche medie della K-Fold Cross Validation**")
            print(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
            print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
            print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
            print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

            if best_cm is not None:
                plt.figure(figsize=(6,6))
                sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix of Best Model")
                plt.show()
        
        return best_model, best_fold_metrics
