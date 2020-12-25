from glob import glob
import logging
import time
import os
import matplotlib.pyplot as plt

import numpy as np
import torch


logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, device, checkpoint_dir, log_dir, verbose=True, verbose_step=1, model=None, optimizer=None, loss_fn=None, lr_scheduler=None, logger=None):
        self.checkpoint_dir = checkpoint_dir
        self.log_path = log_dir + '/trainer.log'
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = 0
        self.best_loss_summary = 10**6
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.logger = logger if logger else logging.getLogger(self.__class__.__name__)
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.log(f'Trainer is ready. Device is {self.device}')

    def fit(self, train_loader, valid_loader, epochs):
        history = {'loss': list(), 'accuracy': list(), 'val_loss': list(), 'val_accuracy': list()}

        for _ in range(epochs):
            t = time.time()
            loss, accuracy = self.train_one_epoch(train_loader)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}/{epochs} loss: {loss:.5f} acc: {accuracy:.5f} time: {(time.time() - t):.5f}')
            self.save(f'{self.checkpoint_dir}/last-checkpoint.bin')

            val_loss, val_acc = self.validation(valid_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}/{epochs} val_loss: {val_loss:.5f} val_acc: {val_acc:.5f} time: {(time.time() - t):.5f}\n')

            if val_loss < self.best_loss_summary:
                self.best_loss_summary = val_loss
                self.save(f'{self.checkpoint_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.checkpoint_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            self.epoch += 1

        plt.figure(figsize=(6, 6), dpi=100)

        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='valid')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Time')
        plt.legend()
        plt.show()

        plt.savefig('loss.jpg')

        plt.plot(history['accuracy'], label='train')
        plt.plot(history['val_accuracy'], label='valid')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Time')
        plt.legend()
        plt.show()

        plt.savefig('accuracy.jpg')

    def train_one_epoch(self, train_loader):
        self.model.train()

        no_steps = len(train_loader.dataset) / train_loader.batch_size
        total = 0.0
        loss_summary = 0.0
        t = time.time()

        for step, batch_sample in enumerate(train_loader):
            self.model.zero_grad()

            images = batch_sample['image'].to(self.device)
            questions = batch_sample['question'].to(self.device)
            answers = batch_sample['answer'].to(self.device)
            output = self.model(images, questions)
            _, answers_ids = torch.max(output, axis=1)
            loss = self.loss_fn(output, answers)

            answers_ids = answers_ids.detach().cpu().numpy()
            answers_ids = answers_ids.reshape(len(answers_ids), 1)
            actual_answers = np.array(batch_sample['actual_answers'])

            # calculate step accuracy
            # acc = min(humans that provided that answer / 3, 1)
            acc_temp = (actual_answers == answers_ids).sum(axis=1) / 3.0
            acc_temp[(acc_temp > 1) == True] = 1
            total += acc_temp.sum()

            loss.backward()

            loss = loss.item()
            loss_summary += loss

            self.optimizer.step()

            if self.verbose:
                if step % self.verbose_step == 0:
                    self.logger.info(
                        f'Step {step}, '
                        f'loss: {loss:.5f}, '
                        f'time: {(time.time() - t):.5f}, '
                        f'\nANSWER IDS\n: {answers_ids}, '
                        f'\nACTUAL\n: {actual_answers}.'
                    )

        # calculate epoch accuracy
        accuracy = total / len(train_loader.dataset)
        # epoch loss
        loss_summary = loss_summary / no_steps

        return loss_summary, accuracy

    def validation(self, valid_loader):
        self.model.eval()

        no_steps = len(valid_loader.dataset) / valid_loader.batch_size
        total = 0.0
        loss_summary = 0.0

        t = time.time()
        for step, batch_sample in enumerate(valid_loader):
            with torch.no_grad():
                images = batch_sample['image'].to(self.device)
                questions = batch_sample['question'].to(self.device)
                answers = batch_sample['answer'].to(self.device)

                output = self.model(images, questions)

                answers_ids, _ = torch.max(output, axis=1)
                loss = self.loss_fn(output, answers)

                answers_ids = answers_ids.detach().cpu().numpy()
                answers_ids = answers_ids.reshape(len(answers_ids), 1)
                actual_answers = np.array(batch_sample['actual_answers'])

                # calculate step accuracy
                acc_temp = (actual_answers == answers_ids).sum(axis=1) / 3.0
                acc_temp[(acc_temp > 1) == True] = 1
                total += acc_temp.sum()

                loss = loss.item()
                loss_summary += loss
                if self.verbose:
                    if step % self.verbose_step == 0:
                        self.logger.info(
                            f'Val Step {step}, '
                            f'loss: {loss:.5f}, '
                            f'time: {(time.time() - t):.5f}, '
                            f'\nANSWER IDS\n: {answers_ids}, '
                            f'\nACTUAL\n: {actual_answers}.'
                        )

        # calculate epoch accuracy
        accuracy = total / len(valid_loader.dataset)
        # epoch loss
        loss_summary = loss_summary / no_steps

        self.lr_scheduler.step(loss_summary)

        return loss_summary, accuracy

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_loss_summary': self.best_loss_summary,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss_summary = checkpoint['best_loss_summary']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.verbose:
            self.logger.info(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')



