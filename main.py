import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tqdm import tqdm
import time


def train(model, args, run_name):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter("logs/" + run_name)
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(p=0.5),
        #ransforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = args.batch_size
    trainset = datasets.CIFAR10(root='./../data', train=True,
                                            download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='./../data', train=False,
                                       download=True, transform=test_transform)
    # create dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    # create optimizer, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr, args.epochs * (len(trainloader) // batch_size))
    criterion = torch.nn.CrossEntropyLoss()

    # for-loop 
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        iterloader = iter(trainloader)
        print("Epoch %d training..." % (epoch + 1))
        total_loss = 0
        total_batches = 0
        for data in tqdm(iterloader):
            global_step += 1
            start_time = time.time()
            inputs, labels = data
            B, H, W, C = inputs.shape
            inputs = inputs.to(device)
            labels = labels.to(device)
            # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()
            # zero the parameter gradients
            outputs = model(inputs)
            # forward
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            total_batches += 1
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            # loss backward
            optimizer.step()
            # optimize
            scheduler.step()
        avg_loss = total_loss / total_batches
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch + 1)
        print("Epoch %d Average Loss: %.3f" % (epoch + 1, avg_loss))
        iterloader = iter(testloader)
        total = 0
        correct = 0
        model.eval()
        print("Epoch %d testing..." % (epoch + 1))
        total_loss = 0
        total_batches = 0
        with torch.no_grad():
            for data in tqdm(iterloader):
                inputs, labels = data
                B, H, W, C = inputs.shape
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_batches += 1
                _, predicted = torch.max(outputs, 1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        avg_loss = total_loss / total_batches
        print("Epoch %d Test Average Loss: %.3f" % (epoch + 1, avg_loss))
        print("Epoch %d Test Accuracy: %.2f %%" % (epoch + 1, acc))
        writer.add_scalar("Acc/epoch", acc, epoch + 1)
        writer.add_scalar("Loss/test_epoch", avg_loss, epoch + 1)
    torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, "checkpoint.pt")

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
        # forwardif load == True:
    if args.ckpt_dir != "" and args.run == "test":
        checkpoint = torch.load(args.ckpt_dir + "/checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
    # create testing dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 16
    testset = datasets.CIFAR10(root='./../data', train=False,
                                       download=True, transform=transform)
    # create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    criterion = torch.nn.CrossEntropyLoss()
    # test
        # forward
    iterloader = iter(testloader)
    total = 0
    correct = 0
    total_loss = 0
    total_batches = 0
    model.eval()
    print("Testing...")
    for data in tqdm(iterloader):
        inputs, labels = data
        B, H, W, C = inputs.shape
        inputs = inputs.reshape(B, H*W*C).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_batches += 1
        _, predicted = torch.max(outputs, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total_batches
    acc = 100 * correct / total
    print("Test Average Loss: %.3f" % (avg_loss))
    print("Test Accuracy: %.2f %%"%(acc))
    return acc, avg_loss

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt_dir", type=str, default="")
    args = parser.parse_args()

    run_name = args.run + "_" + args.model + "_" + args.norm + "_" + str(args.dropout) + "_" + str(args.lr)
    import datetime
    run_date = datetime.datetime.now().strftime('%H%M%S')
    run_name = run_name + '_' + run_date
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "vgg":
        model = VGG(dropout=args.dropout).to(device)
    elif args.model == "resnet":
        model = ResNet().to(device)
    elif args.model == "resnext":
        model = ResNext().to(device)
    else: 
        raise AssertionError
    
    
    if args.run == 'train':
        train(model, args, run_name)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError
    
    # train / test
