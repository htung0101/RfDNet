# Training functions.
# author: ynie
# date: Feb, 2020

from net_utils.utils import LossRecorder, LogBoard
from time import time
log_board = LogBoard()

def train_epoch(cfg, epoch, train_iter, trainer, dataloaders):
    '''
    train by epoch
    :param cfg: configuration file
    :param epoch: epoch id.
    :param trainer: specific trainer for networks
    :param dataloaders: dataloader for training and validation
    :return:
    '''
    for phase in ['train', 'val']:
        dataloader = dataloaders[phase]
        batch_size = cfg.config[phase]['batch_size']
        loss_recorder = LossRecorder(batch_size)
        # set mode
        trainer.net.train(phase == 'train')
        # set subnet mode
        trainer.net.module.set_mode()
        cfg.log_string('-' * 100)
        cfg.log_string('Switch Phase to %s.' % (phase))
        cfg.log_string('-'*100)
        for iter, data in enumerate(dataloader):
            if phase == 'train':
                loss = trainer.train_step(data)
                train_iter += 1
            else:
                loss = trainer.eval_step(data)

            # visualize intermediate results.
            if ((iter + 1) % cfg.config['log']['vis_step']) == 0:
                trainer.visualize_step(phase, epoch, iter + 1, data)

            if ((iter + 1) % 100) == 0: #500
                for loss_name in loss:
                    trainer.writer.add_scalar(f"{loss_name}/{phase}", loss[loss_name], train_iter)

            loss_recorder.update_loss(loss)

            if ((iter + 1) % cfg.config['log']['print_step']) == 0: #10
                cfg.log_string('Process %s: Phase: %s. Epoch %d: %d (%d) /%d. Current loss: %s.' % (cfg.config['runname'], phase, epoch, iter + 1, train_iter, len(dataloader), str(loss)))
                log_board.update(loss, cfg.config['log']['print_step'], phase)

        cfg.log_string('=' * 100)
        for loss_name, loss_value in loss_recorder.loss_recorder.items():
            cfg.log_string('Currently the last %s loss (%s) is: %f' % (phase, loss_name, loss_value.avg))
        cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder, train_iter

def train(cfg, trainer, scheduler, bnm_scheduler, checkpoint, train_loader, val_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param scheduler: scheduler for optimizer
    :param bnm_scheduler: scheduler for batch normalization module
    :param trainer: specific trainer for networks
    :param checkpoint: network weights.
    :param train_loader: dataloader for training
    :param val_loader: dataloader for validation
    :return:
    '''
    start_epoch = scheduler.last_epoch
    total_epochs = cfg.config['train']['epochs']
    min_eval_loss = checkpoint.get('min_loss')
    train_iter = 0

    dataloaders = {'train': train_loader, 'val': val_loader}

    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('-' * 100)
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        trainer.show_lr()
        bnm_scheduler.show_momentum()
        start = time()
        eval_loss_recorder, train_iter = train_epoch(cfg, epoch + 1, train_iter, trainer, dataloaders)
        eval_loss = trainer.eval_loss_parser(eval_loss_recorder)
        scheduler.step(eval_loss)
        bnm_scheduler.step()
        cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).' % (epoch + 1, total_epochs, time()-start))
        # save checkpoint
        checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
        checkpoint.save(f'epoch{epoch}_iter{train_iter}')
        checkpoint.save('last')
        cfg.log_string('Saved the latest checkpoint.')

        if epoch==0 or eval_loss<min_eval_loss:
            checkpoint.save('best')
            min_eval_loss = eval_loss
            cfg.log_string('Saved the best checkpoint.')
            cfg.log_string('=' * 100)
            for loss_name, loss_value in eval_loss_recorder.items():
                cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
            cfg.log_string('=' * 100)