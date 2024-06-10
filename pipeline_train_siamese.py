import argparse
import os
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from src.data.datasets import SiameseAerialPhotography
from src.engine.trainer import fit
import numpy as np
from src.models.networks import ResNet, SiameseNet
from src.losses.losses import ContrastiveLoss
from src.utils.tensorboard_logger import Logger
from src.enities.training_pipeline_params import TrainingConfig, read_training_pipeline_params
from torchvision import transforms

mean, std = 0.1307, 0.3081

def pipeline_train(**kwargs):
    config_file = kwargs['config_file']
    params = read_training_pipeline_params(config_file)
    logger = Logger(model_name=params.model.name, module_name=__name__, data_name=params.short_comment)

    params.training_params.save_to_checkpoint = os.path.join(logger.log_dir, 'checkpoints')
    logger.info(
        f'Создание каталога сохранения чек-поинтов нейронной сети. '
        f'Каталог: {params.training_params.save_to_checkpoint}')
    os.makedirs(params.training_params.save_to_checkpoint, exist_ok=True)
    logger.info(f'Torch version: {torch.__version__}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform =transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((mean,), (std,))
                      ])
    train_dataset = SiameseAerialPhotography(
        params.dataset.path_to_train_data, transform)  # Returns pairs of images and target same/different
    test_dataset = SiameseAerialPhotography(params.dataset.path_to_train_data, transform, split='test')

    logger.info(f"\t\tРазмер обучающего датасета: {len(train_dataset)}")
    logger.info(f"\t\tРазмер тестового датасета: {len(test_dataset)}")

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.training_params.train_batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=params.training_params.test_batch_size,
                                              shuffle=False,
                                              **kwargs)

    margin = 1.
    # embedding_net = EmbeddingNet()
    resnet = ResNet()
    model = SiameseNet(resnet)
    model.to(device)

    loss_fn = ContrastiveLoss(margin)

    optimizer = optim.Adam(model.parameters(), lr=params.training_params.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    log_interval = 100

    fit(train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        params,
        device,
        logger,
        log_interval)

    # torch.save(model.state_dict(), 'embedding_net_weights.pth')

    # train_embeddings_cl, train_labels_cl = extract_embeddings(siamese_train_loader, model)
    # plot_embeddings(train_embeddings_cl, train_labels_cl)
    # val_embeddings_cl, val_labels_cl = extract_embeddings(siamese_test_loader, model)
    # plot_embeddings(val_embeddings_cl, val_labels_cl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_file", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()
    pipeline_train(**vars(args))