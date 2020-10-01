# stdlib
import os
# 3p
import torch
from omegaconf import OmegaConf
# project
from model import GeomFmapNet
from shape_matching_dataset import ShapeMatchingDatasetWrapper
from utils import frobenius_loss


def train(params):
    if torch.cuda.is_available() and not params.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if not os.path.exists(params.savedir):
        os.makedirs(params.savedir)

    # create model
    model = GeomFmapNet(params.n_feat, params.in_grid_size, params.lambda_).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # create dataset
    trainset = ShapeMatchingDatasetWrapper(params, train=True)
    trainloader = trainset.get_dataloader(model.feature_extractor, batch_size=params.batch_size,
                                          shuffle=True, num_workers=params.n_cpu, precompute_multi_scale=True)

    # Training loop
    iterations = 0
    for epoch in range(1, params.n_epochs + 1):
        model.train()
        for i, batch in enumerate(trainloader):
            batch = batch.to(device)

            # do iteration
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                C_est = model(batch)
                loss = frobenius_loss(C_est, batch.C_gt)
                loss.backward()
                optimizer.step()

            # log and save model
            iterations += 1
            if iterations % params.log_interval == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, fmap loss:{loss}")

        if (epoch + 1) % params.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(params.savedir, 'epoch{}.pth'.format(epoch)))


if __name__ == "__main__":
    params = OmegaConf.load("config.yaml")
    train(params)
