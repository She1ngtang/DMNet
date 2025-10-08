from data_provider.data_factory import data_provider
from train_basic import Train_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Forecast(Train_Basic):
    def __init__(self, args):
        super(Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach()
                true = batch_y.detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 使用固定学习率，不再使用调度器
        model_optim = self._select_optimizer()

        # 不需要选择学习率调度器了
        criterion = self._select_criterion()

        # 移除学习率调度器
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    loss = criterion(outputs, batch_y)
                    # total_loss = loss + 0.5 * s_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\tCurrent Learning Rate: {}".format(model_optim.param_groups[0]['lr']))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
      test_data, test_loader = self._get_data(flag='test')

      if test:
          print('loading model')
          self.model.load_state_dict(torch.load(os.path.join('./result/' + setting, 'checkpoint.pth')))

      folder_path = './test_results/' + setting + '/'
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

      self.model.eval()
      preds = []
      trues = []

      with torch.no_grad():
          for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)

              batch_x_mark = batch_x_mark.float().to(self.device)
              batch_y_mark = batch_y_mark.float().to(self.device)

              # For PEMS, batch_x_mark and batch_y_mark are not needed
              batch_x_mark = None
              batch_y_mark = None

              # Decoder input
              if self.args.down_sampling_layers == 0:
                  dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                  dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              else:
                  dec_inp = None

              # Encoder - Decoder
              if self.args.use_amp:
                  with torch.cuda.amp.autocast():
                      if self.args.output_attention:
                          outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                      else:
                          outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
              else:
                  if self.args.output_attention:
                      outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                  else:
                      outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

              # 修复这里：直接使用 outputs，不要使用 outputs[0]
              f_dim = -1 if self.args.features == 'MS' else 0
              outputs = outputs[:, -self.args.pred_len:, f_dim:]
              batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

              pred = outputs.detach().cpu().numpy()
              true = batch_y.detach().cpu().numpy()

              preds.append(pred)
              trues.append(true)

      preds = np.array(preds)
      trues = np.array(trues)

      print('test shape:', preds.shape, trues.shape)

      preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
      trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

      # Inverse transform if needed
      if self.args.data == 'PEMS':
          B, T, C = preds.shape
          preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
          trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

      # Save results
      folder_path = './result/' + setting + '/'
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

      T = preds.shape[1]
      mae_list, rmse_list, mape_list = [], [], []
      for i in range(1,T+1):
        mae, mse, rmse, mape, mspe = metric(preds[:, :i, :], trues[:, :i, :])

        print("Step {}: MAE = {:.5f}, RMSE = {:.5f}, MAPE = {:.5f}".format(i, mae, rmse, mape))
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
      avg_mae = sum(mae_list)/len(mae_list)
      avg_rmse = sum(rmse_list)/len(rmse_list)
      avg_mape = sum(mape_list)/len(mape_list)
      print("Average across {} steps: MAE = {:.5f}, RMSE = {:.5f}, MAPE = {:.5f}".format(T, avg_mae, avg_rmse, avg_mape))