import argparse
from skimage.transform import resize
from DataLoader import DataLoader
from preprocessing import preprocess
import numpy as np
from keras.models import load_model
from losses import bce_dice_loss
from metrics import dice_coef, iou
import tensorflow as tf
from run_length_encoding import mask_to_rle
import pandas as pd

# allow tensorflow to use more memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create parser
ap = argparse.ArgumentParser()
ap.add_argument("-test", "--test", help="path to test data", default="data/test")
ap.add_argument("-height", "--height", type=int, default=256)
ap.add_argument("-width", "--width", type=int, default=256)
ap.add_argument("-model", "--model", default="models/powerfull-unet-resnet-bce-dice-loss.h5")
ap.add_argument("-name", "--name", default="my_predictions")
ap.add_argument("-threshold", "--threshold", type=float, default=0.7)
ap.add_argument("-rle", "--rle", type=int, default=0)
args = vars(ap.parse_args())

# load data
data = DataLoader(test_path=args['test'])
test = data.load_test_data(load_masks=False)

# create list of original shapes
shapes = [test[i].shape for i in range(len(test))]

# preprocess test data
test = np.array([preprocess(image, args['height'], args['width'], True) for image in test])

# load model
model = load_model(args['model'], custom_objects={'bce_dice_loss': bce_dice_loss, "dice_coef": dice_coef, "iou": iou})

# make predictions
predicted = model.predict(test)

# threshold predictions
predicted = (predicted > args['threshold']).astype(np.uint8)

# create list of upsampled test masks
upsampled_masks = []
for i in range(len(predicted)):
    upsampled_masks.append(resize(np.squeeze(predicted[i]),
                                  (shapes[i][0], shapes[i][1]),
                                  mode='constant', preserve_range=True))

# save predictions
np.save("predictions/" + args['name'] + ".npy", np.array(upsampled_masks))

# make run length encoding if we need
if args['rle'] > 0:
    labels = data.load_labels(args['test'])
    print(len(labels))
    print(len(upsampled_masks))

    test_ids, rles = mask_to_rle(upsampled_masks, labels)

    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(f'encoded_predictions/{args["name"]}.csv', index=False)
