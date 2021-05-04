#!/usr/bin/env python3

import argparse
import dataclasses
import pickle
import torch
import torch.utils.data


@dataclasses.dataclass
class ProgramArguments:
    filename : str


def parse_program_arguments():
    parser = argparse.ArgumentParser(
        description="Load a latent-space data-set created from MNIST data")

    parser.add_argument(
        "filename",
        type=str,
        help="filename of the dataset")

    args = parser.parse_args()

    return ProgramArguments(
        filename=args.filename)


class LatentToDigit(torch.utils.data.Dataset):
    """
    Data set containing a mapping from latent space to digits.
    """

    def __init__(self, filename, training=True, max_size=None):
        """
        :param filename: filename of the data set
        :param training: load training data or test data?
        :param max_size: maximum number of elements
        """
        super(LatentToDigit, self).__init__()

        with open(filename, "rb") as fp:
            data = pickle.load(fp)

        data = data["training" if training else "test"]

        data = self._tensorfy(data)

        self.dataset = list(data)
        self.max_size = max_size

    def _tensorfy(self, data):
        for x, index in data:
            x, y = torch.Tensor(x), torch.tensor(index, dtype=torch.long)

            yield x, y

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.dataset))

        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def main():
    args = parse_program_arguments()

    training = LatentToDigit(filename=args.filename, training=True)
    test = LatentToDigit(filename=args.filename, training=False)

    print(f"size of the training data-set: {len(training)}")
    print(f"size of the test data-set: {len(test)}")


if __name__ == "__main__":
    main()
