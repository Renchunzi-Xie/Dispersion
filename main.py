import argparse
from algs.utils import create_alg
from data.utils import build_dataloader
import numpy as np
import time

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--alg', default='standard', type=str)
parser.add_argument('--score', default='id_ood_gap', type=str)

parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--cifar_data_path',
                    default='../datasets/Cifar10', type=str)
parser.add_argument('--cifar_corruption_path',
                    default='../datasets/Cifar10/CIFAR-10-C', type=str)
parser.add_argument('--corruption', default='all', type=str)
parser.add_argument('--severity', default=0, type=int)
parser.add_argument('--dataname', default='cifar10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_samples', default=50000, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_f1_score', default=False, type=bool)

args = vars(parser.parse_args())

import torch
if args["gpu"] is not None:
    device = torch.device(f"cuda:{args['gpu']}")
else:
    device = torch.device('cpu')


def correlation(var1, var2):
    return np.corrcoef(var1, var2)[0, 1]
def correlation2(var1, var2):
    return (np.corrcoef(var1, var2)[0, 1]) ** 2
# spearman
def spearman(var1, var2):
    from scipy import stats
    return stats.spearmanr(var1, var2)

num_class_dict = {
    "cifar10":10,
    "cifar100":100,
    "tinyimagenet":200
}

args["num_classes"] = num_class_dict[args["dataname"]]

if __name__ == "__main__":
    # device
    if "cifar" in args["dataname"]:
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
                           "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
        max_severity = 5
    elif "tinyimagenet" in args["dataname"]:
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                           "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
        max_severity = 5
    else:
        raise TypeError('No relevant corruption list!')

    if args["corruption"] == "all" and args["severity"] == -1:
        scores_list = []
        test_acc_list = []
        time_list = []
        print('alg:{}, dataname:{}, model:{}'.format(args['alg'], args['dataname'], args['arch']))
        for corruption in corruption_list:
            for severity in range(1, max_severity+1):
                args["corruption"] = corruption
                args["severity"] = severity
                # (original x, true labels)
                val_loader = build_dataloader(args['dataname'], args)
                # Define model
                alg_obj = create_alg(args['alg'], val_loader, device, args)
                start_time = time.time()
                scores = alg_obj.evaluate()
                end_time = time.time()
                test_acc = alg_obj.test()
                scores_list.append(float(scores))
                time_list.append(float(end_time - start_time))
                test_acc_list.append(float(test_acc))
                print('corruption:{}, severity:{}, score:{}, test acc:{}'.format(args['corruption'], args['severity'], scores, test_acc))
        mean_score = np.mean(scores_list)
        mean_time = np.mean(time_list)
        print('Mean scores:{}, time:{}'.format(mean_score, mean_time))
        print("Correlation:{}".format(correlation2(scores_list, test_acc_list)))
        print("Spearman:{}".format(spearman(scores_list, test_acc_list).correlation))

    elif args["corruption"] == "all":
        scores_list = []
        test_acc_list = []
        time_list = []
        print('alg:{}, severity:{}, dataname:{}, model:{}'.format(args['alg'], args['severity'],
                                                                                                args['dataname'], args['arch']))
        for corruption in corruption_list:
            args["corruption"] = corruption
            # (original x, true labels)
            val_loader = build_dataloader(args['dataname'], args)
            # Define model
            alg_obj = create_alg(args['alg'], val_loader, device, args)
            start_time = time.time()
            scores = alg_obj.evaluate()
            end_time = time.time()
            test_acc = alg_obj.test()
            scores_list.append(float(scores))
            time_list.append(float(end_time-start_time))
            test_acc_list.append(float(test_acc))
            print('corruption:{}, score:{}, test acc:{}'.format(args['corruption'], scores, test_acc))
        mean_score = np.mean(scores_list)
        mean_time = np.mean(time_list)
        print('Mean scores:{}, time:{}'.format(mean_score, mean_time))
        print("Correlation:{}".format(correlation2(scores_list, test_acc_list)))
        print("Spearman:{}".format(spearman(scores_list, test_acc_list).correlation))
    else:
        val_loader = build_dataloader(args['dataname'], args)
        # Define model
        alg_obj = create_alg(args['alg'], val_loader, device, args)
        start_time = time.time()
        scores = alg_obj.evaluate()
        end_time = time.time()
        time_assumption = float(end_time-start_time)
        test_acc = alg_obj.test()
        print('alg:{}, corruption:{}, severity:{}, dataname:{}, model:{}, scores:{}, test_acc:{}, time:{}'.format(
            args['alg'], args['corruption'], args['severity'],
            args['dataname'], args['arch'], scores, test_acc, time_assumption))







