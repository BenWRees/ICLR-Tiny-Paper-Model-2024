"""
    Training classes.
    Credit to Moor, Michael, et al. "Topological autoencoders." International conference on machine learning. PMLR, 2020.


"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

class TrainingLoop():
    """Training a model using a dataset."""

    def __init__(self, model, dataset, user_item_mat, n_epochs, batch_size, learning_rate,
                 weight_decay=1e-5, device='cuda', callbacks=None):
        """Training of a model using a dataset and the defined callbacks.

        Args:
            model: AutoencoderModel
            dataset: Dataset
            user_item_mat : user item matrix to be trained on
            n_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: List of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.user_item_mat = user_item_mat
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []

    def _execute_callbacks(self, hook, local_variables):
        stop = False
        for callback in self.callbacks:
            # Convert return value to bool --> if callback doesn't return
            # anything we interpret it as False
            stop |= bool(getattr(callback, hook)(**local_variables))
        return stop

    def on_epoch_begin(self, local_variables):
        """Call callbacks before an epoch begins."""
        return self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, local_variables):
        """Call callbacks after an epoch is finished."""
        return self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)

    # pylint: disable=W0641
    def __call__(self):
        """Execute the training loop."""
        print('TRAINING STARTED')
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate

        n_instances = len(dataset)
        # TODO: Currently we drop the last batch as it might not evenly divide
        # the dataset. This is necassary because the surrogate approach does
        # not yet support changes in the batch dimension.
        #WHAT TYPE OF DATA SHOULD WE USE
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)

        n_batches = len(train_loader)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=self.weight_decay)

        epoch = 1
        loss_vals = []
        for epoch in range(1, n_epochs+1) :
            if self.on_epoch_begin(remove_self(locals())):
                break

            for batch, (img,label) in enumerate(train_loader):
                # if self.device == 'cuda':
                #     img = img.cuda(non_blocking=True)

                self.on_batch_begin(remove_self(locals()))
                training_data = self.user_item_mat[:, img].squeeze().permute(1, 0).to(torch.device('cpu'))

                # Set model into training mode and compute loss
                model.train()
                loss, loss_components = model(training_data)

                loss_vals.append(loss.item())

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Call callbacks
                self.on_batch_end(remove_self(locals()))

            if self.on_epoch_end(remove_self(locals())):
                break
        return epoch, loss_vals


def remove_self(dictionary):
    """Remove entry with name 'self' from dictionary.

    This is useful when passing a dictionary created with locals() as kwargs.

    Args:
        dictionary: Dictionary containing 'self' key

    Returns:
        dictionary without 'self' key

    """
    del dictionary['self']
    return dictionary
