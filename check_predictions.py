from DataLoader import DataLoader
import cv2
import argparse
from metrics import dice_coef, iou
import numpy as np
from keras import backend as K

# construct parser
ap = argparse.ArgumentParser()
ap.add_argument('-test', '--test', help="path to test data", default="data/test")
ap.add_argument('-predictions', '--predictions', help="path to predicted masks",
                default="predictions/my_predictions_powerfull-unet-resnet.npy")
ap.add_argument('-metric', "--metric", default="dice")
ap.add_argument('-debug', '--debug', type=int, default=0)
args = vars(ap.parse_args())

# dict of metrics
metrics = {"iou": iou,
           "dice": dice_coef}

# load data and predictions
data = DataLoader(test_path="data/test")
images, masks = data.load_test_data(load_masks=True)
predicted_masks = np.load(args['predictions'], allow_pickle=True)

# normalize masks
masks = masks / 255

# metric
calcul_metrics = np.array([K.get_value(metrics[args['metric']](masks[i], predicted_masks[i])) for i in range(len(masks))])

print(f"Maximum {args['metric']} score:" + str(np.max(calcul_metrics)))
print(f"Minimum {args['metric']} score:" + str(np.min(calcul_metrics)))
print(f"Average {args['metric']} score:" + str(np.mean(calcul_metrics)))

# debug
if args['debug'] > 0:
    for i in range(len(masks)):
        cv2.imshow("Image", images[i])
        cv2.imshow("Mask", masks[i])
        cv2.imshow("Predicted mask", predicted_masks[i])
        print(f"{args['metric']} score:" + str(calcul_metrics[i]))
        cv2.waitKey(0)