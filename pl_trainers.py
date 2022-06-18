from models.models_new import AttentionModel,GRUModel,GRUModel2,AttentionModelConv1d
import torch.optim as optim
import pytorch_lightning as pl
from dataloader.np_loader import NumpyDataDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from utility import amex_metric_mod

def lrfn(cfg, epoch):
    if epoch < cfg.LR_RAMPUP_EPOCHS:
        lr = (cfg.LR_MAX - cfg.LR_START) / cfg.LR_RAMPUP_EPOCHS * epoch + cfg.LR_START
    elif epoch < cfg.LR_RAMPUP_EPOCHS + cfg.LR_SUSTAIN_EPOCHS:
        lr = cfg.LR_MAX
    else:
        decay_total_epochs = cfg.EPOCHS - cfg.LR_RAMPUP_EPOCHS - cfg.LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - cfg.LR_RAMPUP_EPOCHS - cfg.LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (cfg.LR_MAX - cfg.LR_MIN) * cosine_decay + cfg.LR_MIN
    return lr


class ModelTrainer(pl.LightningModule):
    def __init__(self, cfg, X_train, y_train, X_val, y_val, num_workers = 15):
        super().__init__()
        
        # self.model = AttentionModel(cfg.embed_dim, cfg.feat_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout_rate, cfg.num_blocks)
        # self.model = GRUModel2(cfg.embed_dim, cfg.feat_dim, cfg.dropout_rate, cfg.num_layers_rnn)
        self.model = AttentionModelConv1d(cfg.embed_dim, cfg.feat_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout_rate, cfg.num_blocks)
        
        self.l_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        
        self.criterion = nn.BCELoss()
        self.num_workers = num_workers
        
        self.save_preds = {'preds':[], 'target':[]}
        
        self.cfg = cfg

    def forward(self, x_cont):
        return self.model(x_cont)

    def train_dataloader(self):
        train_dataset = NumpyDataDataset(self.X_train, self.y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = NumpyDataDataset(self.X_val, self.y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size,num_workers=self.num_workers)
        return val_loader

    def training_step(self, batch, batch_idx):
        x_cont, y = batch
        outputs = self.forward(x_cont)
        
        loss = self.criterion(outputs, y)
        
        self.log("train/step/loss", loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # print(self.current_epoch)
        pass

    def validation_step(self, batch, batch_idx):
        x_cont, y = batch
        outputs = self.forward(x_cont)
        
        val_loss = self.criterion(outputs, y)
        
        self.log("valid/step/val_loss", val_loss)
        
        self.save_preds['preds'].extend(outputs.tolist())
        self.save_preds['target'].extend(y.tolist())
        return {'val_loss': val_loss}
    
    def validation_epoch_end(self, validation_step_outputs):
        # print(self.save_preds['target'], self.save_preds['preds'])
        score_val = amex_metric_mod(self.save_preds['target'], self.save_preds['preds'])
        print("Eval metric value: {:.3%} Validation Set".format(score_val))
        # print(f"Eval metric (Validation Set){score_val}")
        
        self.log("valid/epoch/amex_metric_mod", score_val)
        
        self.save_preds = {'preds':[], 'target':[]}
        return {'amex_metric_mod': score_val}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        for pg in optimizer.param_groups:
            pg["lr"] = lrfn(self.cfg, epoch)