import datasets
import torch.utils.data


# filename of the data set
DATA_SET = "latent_to_digit.pickle"

# swap the training data-set and the validation data-set?
SWAP_DATA_SETS = False


def load_data(
        batch_size,
        training=True,
        shuffle=True,
        swap=SWAP_DATA_SETS,
        max_size=None,
        filename=None):
    """
    Returns a data loader for the latent-space classification task.

    :param batch_size: batch size of the data loader
    :param training: load training data or validation data
    :param shuffle: shuffle the data?
    :param swap: swap the training and the validation data-set?
    :param max_size: limit the number of elements in the data set
    :param filename: filename of the data set
    :return: data loader containing the data set
    """
    if filename is None:
        filename = DATA_SET

    training = not training if swap else training

    data = datasets.LatentToDigit(
        filename=filename,
        training=training,
        max_size=max_size)

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle)


class Trainer:
    """
    Abstraction of the training process. We want to focus on the major parts.
    """

    def __init__(self, model, loss_function, optimizer, device):
        """
        :param model: model which is optimized
        :param loss_function: loss function used for optimization
        :param optimizer: the optimizer
        :param device: device on which the processing is performed
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, data_loader):
        """
        Training of a single epoch.

        :param data_loader: data loader use to create batches
        """
        total_loss = 0.0
        for i, (xs, ys) in enumerate(data_loader):
            xs, ys = xs.to(self.device), ys.to(self.device)

            pred = self.model(xs)
            loss = self.loss_function(pred, ys)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss

    def validate(self, data_loader, verbose=True):
        """
        Test the model with the validation data.

        :param data_loader: data loader use to create batches
        :param verbose: output status?
        :return: loss, number of correct classifications, number of features,
                 and accuracy
        """
        size = len(data_loader.dataset)
        self.model.eval()
        loss, correct = 0.0, 0

        with torch.no_grad():
            for xs, ys in data_loader:
                xs, ys = xs.to(self.device), ys.to(self.device)
                pred = self.model(xs)
                loss += self.loss_function(pred, ys).item()
                correct += (pred.argmax(1) == ys).type(torch.int).sum().item()

        average_loss = loss / size
        accuracy = correct / size * 100.0

        if verbose:
            print(f"  loss:         {loss}")
            print(f"  average loss: {average_loss}")
            print(f"  correct:      {correct} / {size} ({accuracy:.4}%)")

        return loss, correct, size, accuracy

    def fit(self, num_epochs, training_data, validation_data, verbose=True):
        """
        Fit the model.

        :param num_epochs: train for this number of epochs
        :param training_data: training-data loader
        :param validation_data: test-data loader
        :param verbose: output test status?
        :return: epoch, training loss, validation loss, accuracy
        """
        losses = []

        for epoch in range(num_epochs):
            print("-" * 80)
            print(f"Epoch: {epoch}")

            training_loss = self.train_epoch(
                data_loader=training_data)
            validation_loss = self.validate(
                data_loader=validation_data,
                verbose=verbose)

            losses.append((
                epoch,
                training_loss,
                validation_loss[0],
                validation_loss[3]))

        return losses

    def predict_dataset(self, data):
        """
        :param data: data set which is predicted
        :return: list containing tuples (true label, predicted label)
        """
        self.model.eval()

        matches = []
        with torch.no_grad():
            for xs, ys in data:
                xs = xs.to(self.device)
                prediction = self.model(xs)
                indices = prediction.argmax(1)
                indices = indices.to("cpu")

                matches.extend(
                    (y.item(), idx.item()) for (y, idx) in zip(ys, indices))

        return matches

