import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from DataLoader import DataLoader
from preprocessing import preprocess
from augmentation import image_augmentation
from losses import dice_loss, bce_dice_loss
from metrics import dice_coef, iou
from Unet import Unet
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf

# set seed
seed = 6

# allow tensorflow to use more memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create parser
ap = argparse.ArgumentParser()
ap.add_argument("-index", "--index", help="index of model to save", required=True)
ap.add_argument("-height", "--height", type=int, default=256)
ap.add_argument("-width", "--width", type=int, default=256)
ap.add_argument("-train", "--train", help="path to train data", default="data/train")
ap.add_argument("-batch", "--batch", type=int, default=16)
ap.add_argument("-epochs", "--epochs", type=int, default=50)
ap.add_argument("-loss", "--loss", type=str, default="bce-dice")
ap.add_argument("-resnet", "--resnet", type=int, default=0)
args = vars(ap.parse_args())

# using of residual connection
resnet = "-resnet-" if args['resnet'] > 0 else "-"

# dictionary of losses
losses = {
    "dice": dice_loss,
    "bce": binary_crossentropy,
    "bce-dice": bce_dice_loss}

# load data
data = DataLoader(args['train'])
images, masks = data.load_train_data()

# preprocess_data
images = np.array([preprocess(image, args['height'], args['width'], True) for image in images])
masks = np.array([preprocess(mask, args['height'], args['width'], False) for mask in masks])

# split into train ant validation
X_train, X_valid, y_train, y_valid = train_test_split(images, masks, test_size=0.15, random_state=seed)

# augment data and get generators
train_generartor = image_augmentation(X_train, y_train, seed, args['batch'])

# create, get and compile model
u_net = Unet(args['height'], args['width'], 1, args['resnet'] > 0)
model = u_net.get_model()
model.compile(optimizer=Adam(0.001), loss=losses[args['loss']], metrics=[dice_coef, iou, "acc"])
model.summary()

# create keras callbacks
earlystopper = EarlyStopping(patience=5)
checkpointer = ModelCheckpoint('models/' + args['index'] + resnet + args['loss'] + "-" + "loss" + ".h5", verbose=1,
                               save_best_only=True)

# fit model
history = model.fit(train_generartor,
                    validation_data=(X_valid, y_valid),
                    batch_size=args['batch'],
                    steps_per_epoch=len(X_train) // (args['batch'] * 2),
                    epochs=args['epochs'],
                    verbose=1,
                    callbacks=[earlystopper, checkpointer])

# save model history
np.save("training_history/" + args['index'] + resnet + args['loss'] + "-" + "loss" + "_history.npy", history.history)
