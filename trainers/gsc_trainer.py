#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
#*----------------------------------------------------------------------------*
import os
import sys
import pprint
sys.path.append(os.getcwd())

import argparse
import json
import torch
from torchinfo import summary
from tqdm import tqdm
import wandb
from dataloaders import GSCDataLoader
from models import TCResNet14
from trainers.utils import AverageMeter, EarlyStopping, CheckPoint
from trainers.metrics import accuracy

def run(config_file, in_dir, out_dir):
    print("[GSC Trainer] Config File:", config_file)
    print("[GSC Trainer] Input Directory:", in_dir)
    print("[GSC Trainer] Output Directory:", out_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[GSC Trainer] Training on:", device)

    wandb.init(project='flexnas', entity='embeddedml-edagroup')
    run_name = wandb.run.name
    print("[GSC Trainer] WandB initialized")

    # config
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("[GSC Trainer] Configuration:")
    pprint.pprint(config)
    wandb.config.update(config)

    # data
    train = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'train')
    val = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'valid')
    test = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'test')

    input_size = list(next(iter(train))['data'].shape)
    input_size[0] = 1 # remove batch size
    print("[GSC Trainer] Input size is:", input_size)

    # model
    net = TCResNet14(config['model'])
    net.to(device)
    summary(net, input_size, depth=4)

    # training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    early_stop = EarlyStopping(config['training']['es_patience'], 'max')
    checkpoint = CheckPoint(os.path.join(out_dir, run_name), net, optimizer, 'max')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min', factor=config['training']['lr_factor'],
    #     patience=config['training']['lr_patience'], verbose=True)

    # training loop
    # wandb.watch(net, criterion, log="all", log_freq=10)
    for epoch in range(config['training']['n_epochs']):
      metrics = train_one_epoch(epoch, net, criterion, optimizer, train, val, device)
      # scheduler.step(metrics['val_loss'])
      checkpoint(epoch, metrics['val_acc'])
      if early_stop(metrics['val_acc']):
        break

    # eval and save
    checkpoint.load_best()
    checkpoint.save(os.path.join(out_dir, 'final_best.ckp'))
    test_loss, test_acc = evaluate(net, criterion, test, device)
    test_loss, test_acc = test_loss.get(), test_acc.get()
    print("Test Set Loss:", test_loss)
    print("Test Set Accuracy:", test_acc)
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    wandb.save(os.path.join(out_dir, 'final_best.ckp'))
    wandb.finish()


def run_model(net, datum, target, criterion):
  output = net(datum)
  loss = criterion(output, target)
  return output, loss


def evaluate(net, criterion, data_loader, device, neval_batches = None):
  net.eval()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  with torch.no_grad():
    for batch in data_loader:
      step += 1
      datum, target = batch['data'].to(device), batch['target'].to(device)
      output, loss = run_model(net, datum, target, criterion)
      acc_val = accuracy(output, target)
      avgacc.update(acc_val[0], datum.size(0))
      avgloss.update(loss, datum.size(0))
      if neval_batches is not None and step >= neval_batches:
        return avgloss, avgacc
  return avgloss, avgacc


def train_one_epoch(epoch, net, criterion, optimizer, train, val, device):
  net.train()
  avgacc = AverageMeter('6.2f')
  avgloss = AverageMeter('2.5f')
  step = 0
  with tqdm(total=len(train), unit="batch") as tepoch:
    tepoch.set_description(f"Epoch {epoch}")
    for batch in train:
      step += 1
      tepoch.update(1)
      datum, target = batch['data'].to(device), batch['target'].to(device)
      output, loss = run_model(net, datum, target, criterion)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc_val = accuracy(output, target, topk=(1,))
      avgacc.update(acc_val[0], datum.size(0))
      avgloss.update(loss, datum.size(0))
      if step % 100 == 99:
        tepoch.set_postfix({'loss': avgloss, 'acc': avgacc})
    val_loss, val_acc = evaluate(net, criterion, val, device)
    final_metrics = {
        'loss': avgloss.get(),
        'acc': avgacc.get(),
        'val_loss': val_loss.get(),
        'val_acc': val_acc.get(),
        }
    tepoch.set_postfix(final_metrics)
    tepoch.close()
  wandb.log(final_metrics)
  return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default=None, type=str, help='config file path')
    parser.add_argument('-i', '--in_dir', default=None, type=str, help='input data directory path')
    parser.add_argument('-o', '--out_dir', default=None, type=str, help='output data directory path')
    args = vars(parser.parse_args())
    run(**args)