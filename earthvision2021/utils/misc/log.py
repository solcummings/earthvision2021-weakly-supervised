import logging
import os
import pandas as pd


class TrainLog:
    # set root logger as class variable to run when imported
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    def __init__(self, out_dir,
            log_keys: list[str], save_keys: list[str], save_modes: list[str]):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        self.out_csv = os.path.join(self.out_dir, 'train.csv')
        self.out_log = os.path.join(self.out_dir, 'train.log')

        formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(self.out_log, mode='w+')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.log_keys = log_keys
        self.df = pd.DataFrame([], columns=self.log_keys)
        self.save_keys = save_keys
        self.save_modes = save_modes

        assert len(self.save_keys) == len(self.save_modes)
        self.best_values = [100. if m == 'min' else 0. for m in save_modes]
        self.update_count = 0

    def __call__(self, checkpoint_dict, log_dict):
        self.log(log_dict)
        save_file_list = self.compare_best(log_dict)
        epoch_row = pd.DataFrame(
                [log_dict[c] for c in self.log_keys], index=self.log_keys).T
        self.df = self.df.append(epoch_row)
        self.df.to_csv(self.out_csv, index=False)

        checkpoint_dict = checkpoint_dict | \
                {'best_values': self.best_values, 'update_count': self.update_count}
        return checkpoint_dict, save_file_list

    def log(self, log_dict, indent=['train', 'val']):
        for i in indent:
            keys = [k for k in log_dict.keys() if k[:len(i)] == i]
            self.logger.info(
                    ', '.join(['{} {:.4f}'.format(k, log_dict[k]) for k in keys])
            )

    def compare_best(self, log_dict, patience=15):
        epoch_values = [log_dict[k] for k in self.save_keys]
        epoch_update = False
        save_filename_list = []
        for i, value in enumerate(epoch_values):
            if self.__update_best(self.best_values[i], value, self.save_modes[i]):
                save_filename = 'best_{}.pth'.format(self.save_keys[i])
                self.logger.info('Saving {}'.format(save_filename))
                self.best_values[i] = value
                self.update_count = 0
                epoch_update = True
                save_filename_list.append(save_filename)
        if not epoch_update:
            self.update_count += 1
        if self.update_count == patience:
            self.logger.info('Done Training')
            exit()
        return save_filename_list

    def __update_best(self, best_value, current_value, mode):
        if mode == 'min':
            return True if current_value < best_value else False
        elif mode == 'max':
            # avoid maxing out criteria during early unstable epochs
            return True if 1 > current_value > best_value else False

