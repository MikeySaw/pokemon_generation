import click
import torch
from models.model import ResNet18
from data.make_dataset import corrupted_dataset

from tqdm import tqdm
import wandb
wandb.init(project="mlops",
           name="mlops_toy")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=30, help="number of epochs to train for")
@click.option("--train_batch_size", default=32, help="batch size to use for training")
@click.option("--log_frequency", default=10, help="frequency of logging metrics to wandb")
@click.option("--device", default="cuda", help="device to use for training")
@click.option("--path", default="data/raw/corruptmnist", help="path to the corrupted data")
def train(lr: float, epochs: int, train_batch_size: int, log_frequency: int, device: torch.device, path: str):
    """Train a model on MNIST."""
    model = ResNet18()
    train_set, _ = corrupted_dataset(path=path)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss() 

    model.to(device)

    for i in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for j, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if j % log_frequency == 0:
                wandb.log({"train_loss": loss.item()})
        
        wandb.log({"train_epoch_loss": train_loss / len(train_loader)})
    torch.save(model.state_dict(), "resnet_model.pt")

@click.command()
@click.argument("model_checkpoint")
@click.option("--test_batch_size", default=64, help="batch size to use for testing")
@click.option("--device", default="cuda", help="device to use for testing")
@click.option("--epochs", default=30, help="number of epochs to test for")
def evaluate(model_checkpoint, device: torch.device, epochs: int):
    """Evaluate a trained model."""
    model = torch.load(model_checkpoint)
    _, test_set = corrupted_dataset()

    test_loader = torch.utils.data.DataLoader( 
        test_set, batch_size=64, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(device)
    test_loss = 0
    
    for i in range(epochs):
        model.eval()
        for j, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()

        wandb.log({"test_loss": test_loss / len(test_loader)})


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()