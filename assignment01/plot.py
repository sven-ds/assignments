#!/usr/bin/env python3

import argparse
import autoencoder
import dataclasses
import torch
import utils


@dataclasses.dataclass
class ProgramArguments:
    model : str
    num_images : int
    data_folder : str


def parse_program_arguments():
    parser = argparse.ArgumentParser(
        description="Plot reconstructed images")

    parser.add_argument(
        "model",
        type=str,
        help="path to AutoEncoder model")

    parser.add_argument(
        "--num_images",
        default=10,
        type=int,
        help="number of images to be shown")

    parser.add_argument(
        "--data_folder",
        default="data",
        type=str,
        help="folder where the training and test data is stored")

    args = parser.parse_args()

    return ProgramArguments(
        model=args.model,
        num_images=args.num_images,
        data_folder=args.data_folder)


def main():
    args = parse_program_arguments()
    device = utils.best_device()

    test_data = utils.load_mnist_digits(
        root=args.data_folder,
        training=False)

    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.num_images,
        shuffle=True)

    # load the model
    model = autoencoder.AutoEncoder(latent_size=16)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    utils.plot(
        model=model,
        loader=test_data_loader,
        device=device,
        num_images=args.num_images)


if __name__ == "__main__":
    main()