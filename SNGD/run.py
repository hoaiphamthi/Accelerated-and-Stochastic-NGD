import copy
import numpy as np
import torch

from torch.optim import Adam, SGD
from optimizer import Adsgd, SNGD, Acc_SNGD
from utils import load_data, accuracy_and_loss, save_results, seed_everything


def run(net, args):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    lrs = []

    if args.optimizer in ["AdGD", "SNGD", "SNGDn", "SNGDh"]:
        prev = True
    else:
        prev = False

    if prev:
        prev_net = copy.deepcopy(net)
        prev_net.to(device)
        prev_net.train()
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        optimizer = SGD(net.parameters(), lr=0.01, momentum=0.0)
    elif args.optimizer == "SGDm":
        optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = Adam(net.parameters(), weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer == "AdGD":
        optimizer = Adsgd(net.parameters(), amplifier=args.lr_amplifier, damping=args.lr_damping, weight_decay=args.weight_decay, eps=args.eps)
        prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == "SNGD":
        optimizer = SNGD(net.parameters(), lr=args.lr, eta0=args.eta0, eta1=args.eta1, beta=args.beta, alpha=args.alpha, weight_decay=args.weight_decay)
        prev_optimizer = SNGD(prev_net.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == "SNGDn":
        optimizer = Acc_SNGD(net.parameters(), lr=args.lr, eta0=args.eta0, eta1=args.eta1, alpha=args.alpha, nesterov=True, weight_decay=args.weight_decay)
        prev_optimizer = Acc_SNGD(prev_net.parameters(), weight_decay=args.weight_decay)
    else:
        optimizer = Acc_SNGD(net.parameters(), lr=args.lr, eta0=args.eta0, eta1=args.eta1, alpha=args.alpha, nesterov=False, weight_decay=args.weight_decay)
        prev_optimizer = Acc_SNGD(prev_net.parameters(), weight_decay=args.weight_decay)

    for epoch in range(args.n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if prev:
                prev_optimizer.zero_grad(set_to_none=True)
                prev_outputs = prev_net(inputs)
                prev_loss = criterion(prev_outputs, labels)
                prev_loss.backward()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if prev:
                optimizer.compute_dif_norms(prev_optimizer)
                prev_net.load_state_dict(net.state_dict())
            optimizer.step()

            running_loss += loss.item()
            if (i % 10) == 0:
                if args.noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * args.batch_size / N_train)
                lrs.append(optimizer.param_groups[0]['lr'])

            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * args.batch_size / N_train)

        if not args.noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    if prev:
        del prev_net
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))


if __name__ == "__main__":
    import argparse
    from resnet import ResNet18, DenseNet, LeNet
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument('--lr_amplifier', type=float, default=0.02,
        help='Coefficient alpha for multiplying the stepsize by (1+alpha) (default: 0.02).')
    parser.add_argument('--lr_damping', type=float, default=1.,
        help='Divide the inverse smoothness by damping (default: 1.).')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')
    parser.add_argument('--dataset', type=str, default='mnist', help='cifar10, cifar100, mnist, FashionMNIST')
    parser.add_argument('--net', type=str, default='resnet', help='resnet, densenet, lenet')
    parser.add_argument('--optimizer', type=str, default='SNGDh', help='SGD, SGDm, AdGD, Adam, SNGD, SNGDn, SNGDh')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr')
    parser.add_argument('--eta0', type=float, default=0.2, help='eta0')
    parser.add_argument('--eta1', type=float, default=0.15, help='eta1')
    parser.add_argument('--beta', type=float, default=0, help='beta')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay parameter (default: 0.).')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of passes over the data (default: 128).')
    parser.add_argument('--checkpoint', type=int, default=125, help='Number of passes over the data (default: 128).')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of passes over the data (default: 120).')
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to run the method (default: 1).')
    parser.add_argument('--noisy_train_stat', type=bool, default=False, help='noisy_train_stat')
    parser.add_argument('--output_folder', type=str, default='./Result/',
                        help='Path to the output folder for saving the logs (optional).')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    N_train = 50000
    trainloader, testloader, num_classes = load_data(dataset=args.dataset, batch_size=args.batch_size)
    checkpoint = len(trainloader) // 3 + 1
    
    n_seeds = 1
    max_seed = 424242
    rng = np.random.default_rng(42)
    seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]

    for r, seed in enumerate(seeds):
        print(type(seed))
        seed_everything(int(seed))
        if args.net == 'densenet':
            net = DenseNet(dataset=args.dataset)
        elif args.net == 'lenet':
            net = LeNet(dataset=args.dataset)
        else:
            net = ResNet18(dataset=args.dataset)
        net.to(device)


        losses, test_losses, train_acc, test_acc, it_train, it_test, lrs, grad_norms = run(net=net, args=args)
        method = f'{args.optimizer}_{args.lr}_{args.eta0}_{args.eta1}_{args.beta}_{args.alpha}'
        experiment = f'{args.dataset}_{args.net}_{args.optimizer}_{args.lr}_{args.eta0}_{args.eta1}_{args.beta}_{args.alpha}'
        save_results(losses, test_losses, train_acc, test_acc, it_train, it_test, lrs=lrs,
                 grad_norms=grad_norms, method=method, experiment=experiment, folder=args.output_folder)
