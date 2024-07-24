from torch.utils.data import DataLoader,Subset
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, \
    load_kiiti_intrinsics
from Datasets.tartanTrajFlowDatasetForEvaluater import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir
import pandas as pd


def to_csv(file, filename):
    data = []
    for line in file:
        data.append(line)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--origin-model', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--SubT', action='store_true', default=False,
                        help='SubT test(default:False)')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--savePose', default='',
                        help='position to save result')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name)
    originvo = TartanVO(args.origin_model)

    datastr = 'tartanair'
    if args.euroc:
        datastr = 'euroc'
    elif args.SubT:
        datastr = 'SubT'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr)
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    testDataset = TrajFolderDataset(args.test_dir, posefile=args.pose_file, transform=transform,
                                    focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    val_indices = list(range(900, len(testDataset)))
    val_subset = Subset(testDataset, val_indices)
    testDataloader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)
    motionlist = []
    Originalmotionlist = []
    testname = datastr + '_' + args.model_name.split('.')[0]
    while True:
        try:
            sample = next(testDataiter)
        except StopIteration:
            break

        motions, flow = testvo.test_batch(sample)
        originmotions, originflow = originvo.test_batch(sample)
        motionlist.extend(motions)
        Originalmotionlist.extend(originmotions)
        motionlist = np.array(motionlist)
        Originalmotionlist = np.array(Originalmotionlist)
        motionlist[:, -3:] = Originalmotionlist[:, -3:]
        motionlist = motionlist.tolist()
        Originalmotionlist = Originalmotionlist.tolist()
        poselist = ses2poses_quat(np.array(motionlist))
        to_csv(poselist, args.savePose)


    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True)
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png',
        title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt', results['est_aligned'])
    elif args.pose_file.endswith('.csv'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectorycsv(args.pose_file, poselist, scale=True)
        if datastr == 'euroc':
            print("==> ATE: %.4f" % (results['ate_score']))
        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/' + testname + '.png',
                  title='ATE %.4f' % (results['ate_score']))
        np.savetxt('results/' + testname + '.txt', results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt', poselist)