import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import sys
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
from tqdm import tqdm
from utilities import accuracy, seed_everything
from collections import defaultdict
from TTA import READ, ABPEM


# TTA for the cav-mae-finetuned model
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50'], help='dataset name')
parser.add_argument("--json-root", type=str, default='/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/vgg', help="validation data json")
parser.add_argument("--label-csv", type=str, default='/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/class_labels_indices_vgg.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=50, help="number of classes")
parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
parser.add_argument("--dataset_mean", type=float, default=-5.081, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.4849, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, default=1024, help="the input length in frames")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, help='weight decay')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
# parser.add_argument("--pretrain_path", type=str, default='/xlearning/mouxing/workspace/MM-TTA/egs/kinetics/exp/testmae02-k50-cav-mae-ft-1e-4-2-0.5-1-bs32-ldaFalse-multimodal-fzFalse-h10-a5/models/audio_model_wa.pth', help="pretrained model path")
parser.add_argument("--pretrain_path", type=str, default='/xlearning/mouxing/workspace/TTA/READ/_code_clean/pretrained_model/vgg_65.5.pth', help="pretrained model path")
parser.add_argument("--gpu", type=str, default='7', help="gpu device number")
parser.add_argument("--testmode", type=str, default='multimodal', help="how to test the model")
parser.add_argument('--tta-method', type=str, default='READ', choices=['ABPEM', 'READ', 'Tent', 'SAR', 'None'], help='which TTA method to be used')
parser.add_argument('--corruption-modality', type=str, default='video', choices=['video', 'audio', 'none'], help='which modality to be corrupted')
# parser.add_argument('--data-val', type=str, default='/xlearning/mouxing/workspace/MM-TTA/audioset-processing/data/ks50_test_json_files/gaussian_noise/severity_5.json', help='path to the validation data json')
parser.add_argument('--severity-start', type=int, default=5, help='the start severity of the corruption')
parser.add_argument('--severity-end', type=int, default=5, help='the end severity of the corruption')
parser.add_argument('--output-csv', type=str, default='./outputs/result.csv', help='output csv file')
parser.add_argument('--cache-size', type=int, default=128)
parser.add_argument('--use-cache', action='store_true', default=False)
parser.add_argument('--prime-ent-k', type=int, default=5)
parser.add_argument('--ent-weight', type=float, default=1.0)
parser.add_argument('--attn-weight', type=float, default=1.0)
parser.add_argument('--fix-loss-bal', action='store_true', default=False)
parser.add_argument('--bal-weight', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=5)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

if args.tta_method in ['ABPEM', 'None']:
    TTAPAC = ABPEM
elif args.tta_method in ['READ']:
    TTAPAC = READ
else:
    raise NotImplementedError()

if args.dataset == 'vggsound':
    args.n_class = 309
elif args.dataset == 'ks50':
    args.n_class = 50

if args.corruption_modality == 'video':
    corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression'
    ]
elif args.corruption_modality == 'audio':
    corruption_list = [
    'gaussian_noise',
    'traffic',
    'crowd',
    'rain',
    'thunder',
    'wind'
    ]
elif args.corruption_modality == 'none':
    corruption_list = ['clean']
    args.severity_start = args.severity_end = 0

mean_acc_list = []
std_acc_list = []
for corruption in corruption_list:
    for severity in range(args.severity_start, args.severity_end+1):
        epoch_accs = []

        if args.corruption_modality == 'none':
            data_val = os.path.join(args.json_root, corruption, 'severity_{}.json').format(severity)
        else:
            data_val = os.path.join(args.json_root, args.corruption_modality, '{}', 'severity_{}.json').format(corruption, severity)
        print('===> Now handling: ', data_val)

        for itr in range(1, args.repeat+1):
            seed = int(str(itr)*3)
            seed_everything(seed=seed)
            print("### Seed= {}, Round {} ###".format(seed, itr))

            # all exp in this work is based on 224 * 224 image
            im_res = 224
            val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                              'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

            tta_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

            if args.model == 'cav-mae-ft':
                print('test a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
                va_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
            else:
                raise ValueError('model not supported')

            if args.pretrain_path == 'None':
                warnings.warn("Note no pre-trained models are specified.")
            else:
                # TTA based on a CAV-MAE finetuned model
                mdl_weight = torch.load(args.pretrain_path)
                if not isinstance(va_model, torch.nn.DataParallel):
                    va_model = torch.nn.DataParallel(va_model)
                miss, unexpected = va_model.load_state_dict(mdl_weight, strict=False)
                print('now load cav-mae finetuned weights from ', args.pretrain_path)
                # print(miss, unexpected)
            # exit()
            # evaluate with multiple frames
            if not isinstance(va_model, torch.nn.DataParallel):
                va_model = torch.nn.DataParallel(va_model)

            va_model.cuda()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #######
            print(data_val)
            print(f'use TTA or no?# {args.tta_method}')
            if args.tta_method == 'None':
                adapt_flag = False
            else:
                adapt_flag = True

            if args.tta_method == 'READ' or args.tta_method == 'None' or args.tta_method == 'ABPEM':

                va_model = TTAPAC.configure_model(va_model)

                trainables = [p for p in va_model.parameters() if p.requires_grad]
                print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in va_model.parameters()) / 1e6))
                print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

                params, param_names = TTAPAC.collect_params(va_model)
                args.counter = 0

                optimizer = torch.optim.Adam([{'params': params, 'lr': args.lr}],
                                             weight_decay=args.wd, 
                                             betas=(0.9, 0.999))

                read_model = TTAPAC.READ(va_model, optimizer, device, args)
                cache = ABPEM.Cache(args.cache_size // args.batch_size)
                if args.use_cache:
                    args.cache = cache
                args.dump = defaultdict(list)
                args.callbacks = dict()

                read_model.eval()
                overlap = []
                with torch.no_grad(): 
                    for epoch in range(1):
                        data_bar = tqdm(tta_loader)
                        batch_accs = []

                        for i, (a_input, v_input, labels) in enumerate(data_bar):
                            a_input = a_input.to(device)
                            v_input = v_input.to(device)
                            outputs, loss= read_model((a_input, v_input), adapt_flag=adapt_flag, labels=labels)  # now it infers and adapts!

                            batch_acc = accuracy(outputs[1], labels, topk=(1,))
                            batch_acc = round(batch_acc[0].item(), 2)
                            batch_accs.append(batch_acc)

                            data_bar.set_description(f'Batch#{i}: L0#{loss[0]:.4f}, L1#{loss[1]:.6f}, L2#{loss[2]:.4f}, ACC#{batch_acc:.2f}')

                        epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)
                        epoch_accs.append(epoch_acc)

                        print('Epoch{}: all acc is {}'.format(epoch, epoch_acc))

                        continue


        # mean_value = np.round(np.mean(epoch_accs), 2)
        mean_value = np.round(np.max(epoch_accs), 2)
        std_value = np.round(np.std(epoch_accs), 2)
        mean_acc_list.append(mean_value)
        std_acc_list.append(std_value)
        print('===>{}-{}, mean: {}, std: {}'.format(corruption,severity,mean_value,std_value))
        with open(args.output_csv, 'a') as f:
            f.write('{}, {}, {}, {}\n'.format(corruption, severity, mean_value, std_value))

print('===>{}-{}, mean: {}, std: {}'.format('AVG','-',np.mean(mean_acc_list),np.mean(std_acc_list)))
with open(args.output_csv, 'a') as f:
    f.write('{}, {}, {}, {}\n'.format('AVG','-',np.mean(mean_acc_list),np.mean(std_acc_list)))
