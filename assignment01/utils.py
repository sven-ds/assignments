import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn


CPU, CUDA = "cpu", "cuda"


def devices():
    """
    Collect the available devices.

    :return: collection of available devices
    """
    result = set([CPU])

    if torch.cuda.is_available():
        result.add(CUDA)

    return result


def best_device(available_devices=None):
    """
    Select best/fastest device for learning.

    :param available_devices: devices to select from; if set to none, the
                              available devices are enumerated internally
    :return: best/fastest device for learning
    """
    if available_devices is None:
        available_devices = devices()

    if 0 == len(available_devices):
        raise Exception("no device available")

    # GPU is the best choice we can make
    if CUDA in available_devices:
        return CUDA

    # return one of the available devices
    return next(iter(available_devices))


def load_mnist_digits(root="data", training=True, download=True):
    """
    Load MNIST digits for training and testing.

    :param root: folder of the data
    :param training: load training data?
    :param download: download data if it is not locally available?
    :return: the MNIST dataset
    """
    return torchvision.datasets.MNIST(
        root=root,
        train=training,
        download=download,
        transform=torchvision.transforms.ToTensor())


def plot(model, loader, device, num_images=10):
    """
    Plot MNIST images and their reconstructed counterpart.

    :param model: model used for reconstruction (AutoEncoder)
    :param loader: data loader
    :param device: device on which the calculations are performed
    :param num_images: max number of images
    """
    with torch.no_grad():
        xs, _ = next(iter(loader))
        xs = xs.to(device)
        prediction = model(xs)

    ps = prediction[:num_images]

    fig, ax = plt.subplots(nrows=2, ncols=ps.shape[0])
    xs, ps = xs.to("cpu"), ps.to("cpu")
    for index, (x, p) in enumerate(zip(xs, ps)):
        x = x.reshape((28, 28))
        p = p.reshape((28, 28))

        ax[0, index].imshow(x)
        ax[1, index].imshow(p)

    plt.show()


def plot_training_vs_validation_vs_accuracy(losses, figsize=None):
    """
    Plot a graph with epochs vs training loss and validation loss.

    :param losses: [(epoch, training loss, validation loss, accuracy)]
    :param figsize: figure size of the plot
    """
    epochs, trainings, validations, accuracies = zip(*losses)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # training loss vs. validation loss
    tax = ax[0]
    tax.set_xlabel("epoch")
    tax.set_ylabel("log(training loss)")
    tax.set_yscale("log")
    tax.plot(epochs, trainings, "-", color="red")
    tax.tick_params(axis="y", labelcolor="red")

    vax = tax.twinx()
    vax.set_ylabel("log(validation loss)")
    vax.set_yscale("log")
    vax.plot(epochs, validations, color="blue")
    vax.tick_params(axis="y", labelcolor="blue")

    # training loss vs. accuracy
    tax = ax[1]
    tax.set_xlabel("epoch")
    tax.set_ylabel("training loss")
    tax.plot(epochs, trainings, "-", color="red")
    tax.tick_params(axis="y", labelcolor="red")

    aax = tax.twinx()
    aax.set_ylabel("accuracy")
    aax.plot(epochs, accuracies, color="green")
    aax.tick_params(axis="y", labelcolor="green")

    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix, figsize=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    seaborn.heatmap(matrix, annot=True, ax=ax, fmt="g")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.show()
