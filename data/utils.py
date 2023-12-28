from data.cifar10 import *
from data.cifar100 import *
from data.tinyimagenet import *
import torch


def build_dataloader(dataname, args):
    random_seeds = torch.randint(0, 10000, (2,))
    if args['severity'] == 0:
        seed = 1
        datatype = 'train'
        corruption_type = 'clean'
    else:
        corruption_type = args['corruption']
        seed = random_seeds[1]
        datatype = 'test'
    if dataname == 'cifar10':
        valset = load_cifar10_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'cifar100':
        valset = load_cifar100_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'tinyimagenet':
        valset = load_tinyimagenet(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    else:
        raise Exception("Not Implemented Error")

    valset_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=args['batch_size'],
                                                num_workers = 4,
                                                shuffle=True)
    return valset_loader
