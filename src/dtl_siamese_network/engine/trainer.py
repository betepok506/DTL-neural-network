import torch
import numpy as np
import time
import os
import json
from dtl_siamese_network.enities.training_pipeline_params import TrainingConfig


def fit(train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        params: TrainingConfig,
        device,
        logger,
        log_interval,
        metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    min_val_loss = 1e6
    n_epochs = params.training_params.num_train_epochs
    start_time_training = time.time()
    for epoch in range(start_epoch, n_epochs):

        # Train stage
        start_time_training_epoch = time.time()
        train_loss, metrics = train_epoch(train_loader,
                                          model,
                                          loss_fn,
                                          optimizer,
                                          device,
                                          logger,
                                          log_interval,
                                          metrics)
        end_time_training_epoch = time.time()
        logger.add_scalar("Train/Loss", train_loss, epoch)

        scheduler.step()

        start_time_evaluate_epoch = time.time()
        val_loss, metrics = test_epoch(val_loader,
                                       model,
                                       loss_fn,
                                       device,
                                       metrics)
        end_time_evaluate_epoch = time.time()

        val_loss /= len(val_loader)
        logger.add_scalar("Validate/Loss", val_loss, epoch)

        logger.info(f"Epoch: {epoch + 1}/{n_epochs}\n")
        logger.info(
            f"Train Loss: {train_loss}; Time: {(end_time_training_epoch - start_time_training_epoch):.4f} \n")
        logger.info(
            f"Validation Loss: {val_loss}; Time: {(end_time_evaluate_epoch - start_time_evaluate_epoch):.4f}")

        if min_val_loss > val_loss:
            logger.info(f' Loss Decreasing.. {min_val_loss:.3f} >> {val_loss:.3f}')
            min_val_loss = val_loss

            model_folder = os.path.join(params.training_params.save_to_checkpoint,
                                        f"epoch_{epoch}_{val_loss:.3f}")
            os.makedirs(model_folder, exist_ok=True)

            learning_progress = {'learning_progress': {'epoch': epoch,
                                                       'lr': params.training_params.lr,
                                                       'val_loss': val_loss,
                                                       'train_loss': train_loss}}

            # Сохранение прогресса обучения
            with open(os.path.join(model_folder, 'progress.json'), 'w') as f:
                json.dump(learning_progress, f)

            path_to_save_checkpoint = os.path.join(model_folder, f"checkpoint_{params.model.name}.pth")
            logger.info(f" Save checkpoint to: {path_to_save_checkpoint}")
            torch.save(model, path_to_save_checkpoint)

    end_time_training = time.time()
    logger.info(f'Общее время обучения модели: {(end_time_training - start_time_training):.4f}')


def train_epoch(train_loader,
                model,
                loss_fn,
                optimizer,
                cuda,
                logger,
                log_interval,
                metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        message = ''
        if batch_idx % log_interval == 0:
            logger.info(f"Train: [{batch_idx * len(data[0])}/{len(train_loader.dataset)} "
                        f"({100. * batch_idx / len(train_loader)})] \t Loss: {np.mean(losses):.6f}")
            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))

            # for metric in metrics:
            #     message += '\t{}: {}'.format(metric.name(), metric.value())
            #
            # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader,
               model,
               loss_fn,
               cuda,
               metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
