import os
import numpy as np
import json
import csv
from datetime import datetime
from tqdm import tqdm
import random
import pandas as pd

import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .utils import acc
from src import models
from src.datasets.custom import data_transforms


__all__ = [
    'Plain'
]

class Plain(pl.LightningModule):
    """
    Defines the architecture for training a model using PyTorch Lightning.

    This class inherits from PyTorch Lightning's LightningModule and sets up the model, optimizers,
    and training/validation/testing steps for the training process.
    """

    name = 'Plain'

    def __init__(self, conf, train_class_counts, id_to_labels, **kwargs):
        """
        Initializes the Plain model.

        Args:
            conf: Configuration object with model parameters.
            train_class_counts: Counts of training classes.
            id_to_labels: Mapping from IDs to label names.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.train_class_counts = train_class_counts
        self.id_to_labels = id_to_labels
        self.net = models.__dict__[self.hparams.model_name](num_cls=self.hparams.num_classes, 
                                                            num_layers=self.hparams.num_layers)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing the list of optimizers and the list of learning rate schedulers.
        """
        # Define parameters for the optimizer
        net_optim_params_list = [
            # Optimizer parameters for feature extraction
            {'params': self.net.feature.parameters(),
             'lr': self.hparams.lr_feature,
             'momentum': self.hparams.momentum_feature,
             'weight_decay': self.hparams.weight_decay_feature},
            # Optimizer parameters for the classifier
            {'params': self.net.classifier.parameters(),
             'lr': self.hparams.lr_classifier,
             'momentum': self.hparams.momentum_classifier,
             'weight_decay': self.hparams.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        optimizer = torch.optim.SGD(net_optim_params_list)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)   
        return [optimizer], [scheduler]

    def on_train_start(self):
        """
        Hook function called at the start of training. Initializes best accuracy and the network.
        """
        self.best_acc = 0
        self.net.feat_init()
        self.net.setup_criteria()
        self.epoch_history = []
        self._train_epoch_losses = []
        self._train_epoch_preds = []
        self._train_epoch_labels = []
        self._val_epoch_losses = []
        self._val_epoch_preds = []
        self._val_epoch_labels = []

    def on_train_epoch_start(self):
        """Reset train epoch accumulators."""
        self._train_epoch_losses = []
        self._train_epoch_preds = []
        self._train_epoch_labels = []

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            Tensor: The loss for the current training step.
        """
        data, label_ids = batch[0], batch[1]
        
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        # Calculate loss
        loss = self.net.criterion_cls(logits, label_ids)
        preds = logits.argmax(dim=1)

        self._train_epoch_losses.append(loss.detach().cpu().item())
        self._train_epoch_preds.append(preds.detach().cpu().numpy())
        self._train_epoch_labels.append(label_ids.detach().cpu().numpy())

        self.log("train_loss", loss)
        
        return loss

    def on_validation_start(self):
        """
        Hook function called at the start of validation. Initializes storage for validation outputs.
        """
        self.val_st_outs = []
        self._val_epoch_losses = []
        self._val_epoch_preds = []
        self._val_epoch_labels = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.
        """
        data, label_ids = batch[0], batch[1]
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        loss = self.net.criterion_cls(logits, label_ids)
        preds = logits.argmax(dim=1)

        self._val_epoch_losses.append(loss.detach().cpu().item())
        self._val_epoch_preds.append(preds.detach().cpu().numpy())
        self._val_epoch_labels.append(label_ids.detach().cpu().numpy())
        
        self.val_st_outs.append((preds.detach().cpu().numpy(),
                                 label_ids.detach().cpu().numpy()))

    def on_validation_epoch_end(self):
        """
        Hook function called at the end of the validation epoch. Aggregates and logs validation results.
        """
        total_preds = np.concatenate([x[0] for x in self.val_st_outs], axis=0)
        total_label_ids = np.concatenate([x[1] for x in self.val_st_outs], axis=0)
        train_preds = np.concatenate(self._train_epoch_preds, axis=0)
        train_labels = np.concatenate(self._train_epoch_labels, axis=0)
        val_preds = np.concatenate(self._val_epoch_preds, axis=0)
        val_labels = np.concatenate(self._val_epoch_labels, axis=0)

        _, _, train_mic_acc = acc(train_preds, train_labels)
        _, _, val_mic_acc = acc(val_preds, val_labels)

        self.epoch_history.append({
            'epoch': int(self.current_epoch),
            'train_loss': float(np.mean(self._train_epoch_losses)),
            'val_loss': float(np.mean(self._val_epoch_losses)),
            'train_acc': float(train_mic_acc * 100),
            'val_acc': float(val_mic_acc * 100)
        })

        self.eval_logging(total_preds, total_label_ids)

    def on_fit_end(self):
        """Write one row per epoch to loss_accuracy.csv in the logger directory."""
        if not self.epoch_history or self.logger is None:
            return

        log_dir = self._get_artifact_dir()
        if log_dir is None:
            return

        self._export_split_results(split='train', output_dir=log_dir)
        self._export_split_results(split='test', output_dir=log_dir)

    def _get_artifact_dir(self):
        log_dir = getattr(self.logger, 'log_dir', None)
        if not log_dir:
            save_dir = getattr(self.logger, 'save_dir', None)
            name = getattr(self.logger, 'name', '')
            version = getattr(self.logger, 'version', '')
            if save_dir is None:
                return None
            log_dir = os.path.join(save_dir, str(name), f'version_{version}')

        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, 'loss_accuracy.csv')
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(self.epoch_history, key=lambda x: x['epoch']):
                writer.writerow(row)

        return log_dir

    def _get_split_dataset(self, split):
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule is None:
            raise RuntimeError('Training datamodule is not available for export.')

        return datamodule.ds(
            rootdir=datamodule.conf.dataset_root,
            dset=split,
            transform=data_transforms['val'],
            conf=datamodule.conf,
        )

    def _export_split_results(self, split, output_dir):
        dataset = self._get_split_dataset(split)
        datamodule = self.trainer.datamodule
        configured_batch_size = int(getattr(datamodule.conf, 'export_batch_size', datamodule.conf.batch_size))
        configured_num_workers = int(getattr(datamodule.conf, 'export_num_workers', datamodule.conf.num_workers))
        export_batch_size = min(max(1, configured_batch_size), max(1, len(dataset)))
        export_num_workers = max(0, configured_num_workers)
        dataloader = DataLoader(
            dataset,
            batch_size=export_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=export_num_workers,
            drop_last=False,
        )

        class_ids = list(range(self.hparams.num_classes))
        class_names = [self.id_to_labels.get(class_id, str(class_id)) for class_id in class_ids]

        rows = []
        y_true = []
        y_pred = []

        was_training = self.training
        export_device = getattr(self.trainer.strategy, 'root_device', self.device)
        self.to(export_device)
        self.eval()

        print(
            f'Exporting {split} predictions and confusion matrix... '
            f'({len(dataset)} images, {len(dataloader)} batches, '
            f'batch_size={export_batch_size}, workers={export_num_workers}, '
            f'device={export_device})'
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Export {split}', leave=False):
                data, label_ids, labels, file_ids = batch
                data = data.to(export_device, non_blocking=True)

                feats = self.net.feature(data)
                logits = self.net.classifier(feats)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                probs_np = probs.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
                label_ids_np = label_ids.detach().cpu().numpy()

                for index in range(len(file_ids)):
                    row = {
                        'path': os.path.basename(str(file_ids[index])),
                        'label': str(labels[index]),
                        'classification': int(label_ids_np[index]),
                    }

                    for class_id in class_ids:
                        row[f'prob_class_{class_id}'] = float(probs_np[index, class_id])

                    rows.append(row)
                    y_true.append(int(label_ids_np[index]))
                    y_pred.append(int(preds_np[index]))

        if was_training:
            self.train()

        predictions_path = os.path.join(output_dir, f'{split}_predictions.csv')
        predictions_df = pd.DataFrame(rows)
        predictions_columns = ['path', 'label', 'classification'] + [f'prob_class_{class_id}' for class_id in class_ids]
        predictions_df = predictions_df[predictions_columns]
        predictions_df.to_csv(predictions_path, index=False)

        confusion = confusion_matrix(y_true, y_pred, labels=class_ids)
        confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
        confusion_path = os.path.join(output_dir, f'{split}_confusion_matrix.csv')
        confusion_df.to_csv(confusion_path, index_label='true_class')
        print(f'Saved {split} predictions to {predictions_path}')
        print(f'Saved {split} confusion matrix to {confusion_path}')

    def on_test_start(self):
        """
        Hook function called at the start of testing. Initializes storage for test outputs.
        """
        self.te_st_outs = []

    def test_step(self, batch, batch_idx):
        """
        Test step for each batch.

        Args:
            batch: The current batch of data, including metadata.
            batch_idx: The index of the current batch.
        """
        data, label_ids, labels, file_ids = batch
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)
        
        self.te_st_outs.append((preds.detach().cpu().numpy(),
                               label_ids.detach().cpu().numpy(),
                               feats.detach().cpu().numpy(),
                               logits.detach().cpu().numpy(), 
                               labels, file_ids 
                               ))
    

    def on_test_epoch_end(self):
        """
        Hook function called at the end of the test epoch. Aggregates and logs test results, and saves output.
        """
        # Concatenate outputs from all test steps
        total_preds = np.concatenate([x[0] for x in self.te_st_outs], axis=0)
        total_label_ids = np.concatenate([x[1] for x in self.te_st_outs], axis=0)
        total_feats = np.concatenate([x[2] for x in self.te_st_outs], axis=0)
        total_logits = np.concatenate([x[3] for x in self.te_st_outs], axis=0)
        total_labels = np.concatenate([x[4] for x in self.te_st_outs], axis=0)
        total_file_ids = np.concatenate([x[5] for x in self.te_st_outs], axis=0)

        # Calculate the metrics and save the output
        self.eval_logging(total_preds[total_label_ids != -1],
                          total_label_ids[total_label_ids != -1],
                          print_class_acc=False)

        output_path = self.hparams.evaluate.replace('.ckpt', 'eval.npz') 
        np.savez(output_path, preds=total_preds, label_ids=total_label_ids, feats=total_feats,
                 logits=total_logits, labels=total_labels, file_ids=total_file_ids)  
        print('Test output saved to {}.'.format(output_path))

    def on_predict_start(self):
        """
        Hook function called at the start of prediction. Initializes storage for prediction outputs.
        """
        self.pr_st_outs = []

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for each batch.

        Args:
            batch: The current batch of data, including metadata.
            batch_idx: The index of the current batch.
        """
        data, file_ids = batch
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1).max(dim=1)[0]
        
        self.pr_st_outs.append((preds.detach().cpu().numpy(),
                                feats.detach().cpu().numpy(),
                                logits.detach().cpu().numpy(), 
                                probs.detach().cpu().numpy(),
                                file_ids 
                                ))
    

    def on_predict_epoch_end(self):
        """
        Hook function called at the end of the predict epoch. Aggregates and saves prediction outputs.
        """
        # Concatenate outputs from all predict steps
        total_preds = np.concatenate([x[0] for x in self.pr_st_outs], axis=0)
        total_feats = np.concatenate([x[1] for x in self.pr_st_outs], axis=0)
        total_logits = np.concatenate([x[2] for x in self.pr_st_outs], axis=0)
        total_probs = np.concatenate([x[3] for x in self.pr_st_outs], axis=0)
        total_file_ids = np.concatenate([x[4] for x in self.pr_st_outs], axis=0)

        json_output = []
        for i in range(len(total_preds)):
            json_output.append({
                "marker_id": "",
                "survey_pic_id": total_file_ids[i],
                "marker_confidence": float(total_probs[i]),
                "marker_gear_type": "ghostnet" if total_preds[i] == 1 else "neg",
                "marker_bounding_polygon": "",
                "marker_status": "unverified",
                "marker_ai_model": ""
            })

        output_path_full = self.hparams.evaluate.replace('.ckpt', '_predict.npz') 
        np.savez(output_path_full, preds=total_preds, feats=total_feats,
                 logits=total_logits, file_ids=total_file_ids)  
        print('Predict output saved to {}.'.format(output_path_full))

        output_path_json = self.hparams.evaluate.replace('.ckpt', '_predict.json') 
        json.dump(json_output, open(output_path_json, 'w'))
        print('Predict output json saved to {}.'.format(output_path_json))


    def eval_logging(self, preds, labels, print_class_acc=False):
        """
        Logs evaluation metrics such as accuracy.

        Args:
            preds: Predictions from the model.
            labels: Ground truth labels.
            print_class_acc (bool): Flag to print class-wise accuracy.
        """
        class_acc, mac_acc, mic_acc = acc(preds, labels)
        unique_eval_labels = np.unique(labels)

        self.log("valid_mac_acc", mac_acc * 100)
        self.log("valid_mic_acc", mic_acc * 100)

        if print_class_acc:

            if self.train_class_counts:
                acc_list = [(class_acc[i], unique_eval_labels[i],
                             self.id_to_labels[unique_eval_labels[i]],
                             self.train_class_counts[unique_eval_labels[i]])
                             for i in range(len(class_acc))]

                print('\n')
                for i in range(len(class_acc)):
                    info = '{:>20} ({:<3}, tr {:>3}) Acc: '.format(acc_list[i][2],
                                                                   acc_list[i][1],
                                                                   acc_list[i][3])
                    info += '{:.2f}'.format(acc_list[i][0] * 100)
                    print(info)
            else:
                acc_list = [(class_acc[i], unique_eval_labels[i],
                             self.id_to_labels[unique_eval_labels[i]])
                             for i in range(len(class_acc))]

                print('\n')
                for i in range(len(class_acc)):
                    info = '{:>20} ({:<3}) Acc: '.format(acc_list[i][2], acc_list[i][1])
                    info += '{:.2f}'.format(acc_list[i][0] * 100)
                    print(info)
