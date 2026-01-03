# client.py
import flwr as fl
import torch
from train_utils import (
    build_model, build_optimizer, build_scheduler, build_criterion,
    get_client_dataloaders, train_one_epoch, evaluate_model, get_device, get_weights, set_weights
)
import warnings

device = get_device()

class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_df, val_df, root_dir, pos_weights):
        self.client_id = client_id
        self.model = build_model().to(device)
        self.optimizer = build_optimizer(self.model)
        self.scheduler = build_scheduler(self.optimizer)
        self.criterion = build_criterion(pos_weights)
        self.train_loader, self.val_loader = get_client_dataloaders(
            client_id, train_df, val_df, root_dir
        )

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        warnings.filterwarnings("ignore", message="No positive class found")
        set_weights(self.model, parameters)
        
        train_loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, device)
        print(f"[Client {self.client_id}] Finished training, loss={train_loss}")
        
        self.scheduler.step()
        val_loss, f1_micro, f1_macro, prec_micro, rec_micro, ap_micro, ap_macro, _, _ = evaluate_model(
            self.model, self.val_loader, self.criterion, device
        )
        
        metrics = {
            "val_loss": val_loss,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "ap_micro": ap_micro,
            "ap_macro": ap_macro,
        }

        return get_weights(self.model), len(self.train_loader.dataset), metrics
    
        for epoch in range(start_epoch, num_epochs):
            print(f"\n🌍 Epoch {epoch+1}/{num_epochs}")

            start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, f1_micro, f1_macro, prec_micro, rec_micro, ap_micro, ap_macro, y_true, y_pred = evaluate_model(
            model, val_loader, criterion, device
        )

            scheduler.step()
            epoch_time = time.time() - start_time

            # --- Store metrics ---
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # --- Store metrics ---
            val_micro_f1s.append(f1_micro)
            val_macro_f1s.append(f1_macro)

            # print(f"⏱️ Epoch time: {epoch_time:.2f}s")
            # print(f"Val Loss: {val_loss:.4f}")
            # print(f"F1 (micro): {f1_micro:.4f} | F1 (macro): {f1_macro:.4f}")
            # print(f"Precision (micro): {prec_micro:.4f} | Recall (micro): {rec_micro:.4f}")
            # print(f"AP (micro): {ap_micro:.4f} | AP (macro): {ap_macro:.4f}")

            # --- Save best model based on micro-F1 ---
            if f1_micro > best_val_f1:
                best_val_f1 = f1_micro
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1
                }, checkpoint_path)
                print(f"✅ Saved new best model (Micro-F1={best_val_f1:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"⚠️ No improvement. Patience counter: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

            # --- Log stats for this epoch ---
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "precision_micro": prec_micro,
                "recall_micro": rec_micro,
                "ap_micro": ap_micro,
                "ap_macro": ap_macro,
                "epoch_time_sec": epoch_time,
                "best_val_f1": best_val_f1
            }
            history.append(epoch_log)

            # Save running logs after each epoch
            with open(log_path, "w") as f:
                json.dump(history, f, indent=4)

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        val_loss, f1_micro, f1_macro, prec_micro, rec_micro, ap_micro, ap_macro, _, _ = evaluate_model(
            self.model, self.val_loader, self.criterion, device
        )
        metrics = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "ap_micro": ap_micro,
            "ap_macro": ap_macro,
        }
        return val_loss, len(self.val_loader.dataset), metrics


# if __name__ == "__main__":
#     # Example: run client 0
#     fl.client.start_numpy_client(server_address="0.0.0.0:8080",
#                                  client=FLClient(client_id=0,
#                                                  df_loaded=df_loaded,
#                                                  root_dir=root_dir,
#                                                  band_means=band_means,
#                                                  band_stds=band_stds,
#                                                  pos_weights=pos_weights))
