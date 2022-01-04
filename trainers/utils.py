import os
import torch

class AverageMeter(object):
  """Computes and stores the average and current value of a metric"""
  def __init__(self, fmt='f', name='meter'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def get(self):
    return float(self.avg)

  def __str__(self):
    fmtstr = '{:' + self.fmt + '}'
    return fmtstr.format(float(self.avg))

class EarlyStopping():
  """
  stop the training when the loss does not improve.
  """
  def __init__(self, patience=5, mode='min'):
    if mode not in ['min', 'max']:
      raise ValueError("Early-stopping mode not supported")
    self.patience = patience
    self.mode = mode
    self.counter = 0
    self.best_val = None

  def __call__(self, val):
    val = float(val)
    if self.best_val == None:
      self.best_val = val
    elif self.mode == 'min' and val < self.best_val:
      self.best_val = val
      self.counter = 0
    elif self.mode == 'max' and val > self.best_val:
      self.best_val = val
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        print("Early Stopping!")
        return True
    return False

class CheckPoint():
  """
  save/load a checkpoint based on a metric
  """
  def __init__(self, dir, net, optimizer, mode='min', fmt='ck_{epoch:03d}.pt'):
    if mode not in ['min', 'max']:
      raise ValueError("Early-stopping mode not supported")
    if not os.path.exists(dir):
      os.makedirs(dir)
    self.dir = dir
    self.mode = mode
    self.format = fmt
    self.net = net
    self.optimizer = optimizer
    self.val = None
    self.epoch = None
    self.best_path = None
  
  def __call__(self, epoch, val):
    val = float(val)
    if self.val == None:
      self.update_and_save(epoch, val)
    elif self.mode == 'min' and val < self.val:
      self.update_and_save(epoch, val)
    elif self.mode == 'max' and val > self.val:
      self.update_and_save(epoch, val)

  def update_and_save(self, epoch, val):
    self.epoch = epoch
    self.val = val
    self.update_best_path()
    self.save()

  def update_best_path(self):
    self.best_path = os.path.join(self.dir, self.format.format(**self.__dict__))

  def save(self, path=None):
    if path is None:
      path = self.best_path
    torch.save({
              'epoch': self.epoch,
              'model_state_dict': self.net.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              'val': self.val,
              }, path)
    
  def load_best(self):
    if self.best_path is None:
      raise FileNotFoundError("Best path not set!")
    self.load(self.best_path)

  def load(self, path):
    checkpoint = torch.load(path)
    self.net.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])