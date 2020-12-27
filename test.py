#from progress.bar import Bar as ProgressBar
#import time

#progressbar1 = ProgressBar('Training', max=100)

#for i in range(0, 100):
#    progressbar1.next()
#    time.sleep(.1)

#    if i % 20 == 0:
#        print()
#        progressbar2 = ProgressBar('Evaluation', max=50)
        
#        for j in range(0, 50):
#            progressbar2.next()
#            time.sleep(.1)
#        progressbar2.finish()
#        print("\033[A")
#progressbar1.finish()




#from tqdm.auto import tqdm
#import time

#bar_format = "{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}"

#with tqdm(total=100, desc="TRAINING", colour='green', position=0, leave=False, bar_format=bar_format) as p1:
#    for i in range(0, 100):
#        time.sleep(.1)

#        p1.set_postfix({"Postfix: ": str(i)})
#        p1.update()
        
#        #p1.set_postfix_str("Loss: %d" % i)

#        if i % 20 == 0:
#            for j in tqdm(range(0, 50), desc="VALIDATION", colour='yellow', position=1, leave=False, bar_format=bar_format):
#                time.sleep(.1)

#            tqdm.write('')
#            tqdm.write('')

#        p1.write("Epoch: %d" % i)

#from torch.utils import tensorboard as tb

#w=tb.SummaryWriter('logs/testing')

#w.add_text("INFO", "Information1", global_step=0)
#w.add_text("INFO", "Information2", global_step=0)

#w.close()




#import torch as t
#import torchvision as tv
#import numpy as np
#from PIL import Image
#from datasets.Cityscapes import settings as cityscapes_settings

#dataset = tv.datasets.Cityscapes('./datasets/Cityscapes/data', target_type='semantic')
##loader = t.utils.data.DataLoader(dataset, batch_size=4)

#min_,max_=(1000, 0)
#for i in range(len(dataset)):
#    i,s = dataset[i]

#    min_=min(min_, np.array(s).min())
#    max_=max(max_, np.array(s).max())

#    def t(x):
#        print(x)
#        return 0
#    picnew = s.point(lambda x: cityscapes_settings.LABEL_MAPPING_DICT[x]) #Image.eval(s, t)

#print('min: {}, max: {}'.format(min_, max_))




#import argparse

#parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Segmantic Segmentation' CVPR 2020 paper.")
#command_parser = parser.add_subparsers(title='commands', dest='command', required=True)

## Training commands
#train_parser = command_parser.add_parser('train', help='Train model for different stages')
#train_parser.add_argument('--resume_weight', default=None, type=str, help="Resume training with given weight file")
#train_parser.add_argument('--device', default='gpu', type=str.lower, choices=['cpu', 'gpu'], help="Device to create model in")
#train_parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loader")
#train_parser.add_argument('--val_interval', default=10, type=int, help="Epoch intervals after which to perform validation")
#train_parser.add_argument('--autosave_interval', default=5, type=int, help="Epoch intervals to auto save weights after in training")
#train_parser.add_argument('--autosave_history', default=5, type=int, help="Number of latest autosaved weights to keep while deleting old ones, 0 to disable autosave")
#train_parser.add_argument('--batch_size', default=4, type=int, help="Batch size to use for training and testing")
#train_parser.add_argument('--epochs', type=int, help="Number of epochs to train")
#train_parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
#train_parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
#train_parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for SGD")
#train_parser.add_argument('--poly_power', type=float, default=0.9, help="Power for poly learning rate strategy")
#train_parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
#train_parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
#train_parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
#train_parser.add_argument('--description', type=str, default='', help="Description of experiment to be saved in 'params.txt' with given commandline parameters")

## Testing commands
#test_parser = command_parser.add_parser('test', help='Test model')
#test_parser.add_argument('--out', default=None, type=str, help="Model weights")

#args = parser.parse_args()

#print(type(args))
#print(args)
#print(dir(args))
#print(args.__dict__)



#import termcolor

#def FATAL(text):
#    return termcolor.colored("FATAL: " + text, 'red', attrs=['reverse', 'blink'])

#print(FATAL("Something went wrong"))




from PIL import Image

img = Image.open('./datasets/Cityscapes/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')

org_size = img.size    # CAUTION: For Image, size is (W, H) order in contrast to (H, W) for Tensor
scale_factor = 3.5

crop_width = int(1.0 / scale_factor * org_size[0])
crop_height = int(1.0 / scale_factor * org_size[1])
crop_x = (org_size[0] - crop_width) // 2
crop_y = (org_size[1] - crop_height) // 2
crop_box = [crop_x,\
            crop_y,\
            crop_x+crop_width,\
            crop_y+crop_height]
print(crop_box)

img = img.resize(size=org_size, resample=Image.BILINEAR, box=crop_box)
img.save('out.png')