# client.py
from datetime import datetime
import flwr as fl
import json
import os
import time
from train_utils import (
    build_model, build_optimizer, build_scheduler, build_criterion,
    get_client_dataloaders, train_one_epoch, evaluate_model, get_device, get_weights, set_weights
)
import warnings
from web3 import Web3

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
        
        # Connect to Ganache
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        assert self.w3.is_connected(), "Web3 not connected to Ganache"

        self.account = self.w3.eth.accounts[0]  # use first Ganache account

        # Load ABI and contract address from Truffle build
        with open("build/contracts/TrainingLogs.json") as f:
            contract_json = json.load(f)
        abi = contract_json["abi"]

        address = contract_json["networks"]["1766572888649"]["address"]

        self.contract = self.w3.eth.contract(address=address, abi=abi)


    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        warnings.filterwarnings("ignore", message="No positive class found")
        set_weights(self.model, parameters)
        
        NUM_EPOCHS = 5
        HOME_DIR = os.getcwd()
        base_log_dir = os.path.join(HOME_DIR, "training_logs")

        # Create timestamp folder once per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(base_log_dir, exist_ok=True)

        # Single log file per run, named with timestamp
        log_path = os.path.join(base_log_dir, "fed_training_log.jsonl")

        train_losses, val_losses = [], []
        val_micro_f1s, val_macro_f1s = [], []
        # history = []
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n🌍 Epoch {epoch+1}/{NUM_EPOCHS}")

            start_time = time.time()
            train_loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, device)
            print(f"[Client {self.client_id}] Finished training, loss={train_loss}")
            
            self.scheduler.step()

            val_loss, f1_micro, f1_macro, prec_micro, rec_micro, ap_micro, ap_macro, _, _ = evaluate_model(
                self.model, self.val_loader, self.criterion, device
            )
            epoch_time = time.time() - start_time
            print(f"[Client {self.client_id}] Finished validation, loss={val_loss}")

            # --- Store metrics ---
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_micro_f1s.append(f1_micro)
            val_macro_f1s.append(f1_macro)

            # --- Log stats for this epoch ---
            epoch_log = {
                "client_id": self.client_id,
                # "round": config.get("server_round", 0),
                "timestamp": datetime.now().isoformat(),
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
            }
            metrics = {
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "f1_micro": val_micro_f1s[-1],
                "f1_macro": val_macro_f1s[-1],
            }
            
            # history.append(epoch_log)

            # Save running logs after each epoch (append to same file)
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

            # --- Store on chain ---
            metrics_struct = (
                int(train_loss * 1e6),
                int(val_loss * 1e6),
                int(f1_micro * 1e6),
                int(f1_macro * 1e6),
                int(prec_micro * 1e6),
                int(rec_micro * 1e6),
                int(ap_micro * 1e6),
                int(ap_macro * 1e6),
            )

            tx_hash = self.contract.functions.storeLog(
                int(self.client_id),
                epoch_log["timestamp"],
                epoch_log["epoch"],
                metrics_struct,
                int(epoch_time)
            ).transact({'from': self.account,
                        'gas': 10**9})

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"[Client {self.client_id}] Log stored on chain in block {receipt.blockNumber}")

        return get_weights(self.model), len(self.train_loader.dataset), metrics

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