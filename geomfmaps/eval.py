# stdlib
import os
# 3p
import torch
from omegaconf import OmegaConf
# project
from model import GeomFmapNet
from shape_matching_dataset import ShapeMatchingDatasetWrapper


def eval_model(model_path, params):
    if torch.cuda.is_available() and not params.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if not os.path.exists(params.evaldir):
        os.makedirs(params.evaldir)

    # create model
    model = GeomFmapNet(params.n_feat, params.in_grid_size, params.lambda_).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # create dataset
    testset = ShapeMatchingDatasetWrapper(params, train=False)
    testloader = testset.get_dataloader(model.feature_extractor, batch_size=1,
                                        shuffle=False, num_workers=params.n_cpu, precompute_multi_scale=True)

    to_save = []
    used_names, combinations = testloader._dataset.used_names, testloader._dataset.combinations
    for i, batch in enumerate(testloader):
        batch = batch.to(device)
        with torch.set_grad_enabled(False):
            C_est = model(batch).t()

        # save
        target, source = used_names[combinations[i][0]], used_names[combinations[i][1]]
        to_save.append({'C_est': C_est, "source": source, "target": target})

    torch.save(to_save, os.path.join(params.evaldir, "fmaps.pt"))


if __name__ == "__main__":
    params = OmegaConf.load("config.yaml")
    PATH = "path/to/model.pt"

    eval_model(PATH, params)