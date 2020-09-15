from keras.preprocessing import image


def image_augmentation(X_train, y_train, seed, batch_size):
    data_generator_args = dict(rotation_range=45.,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.3,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='reflect')

    # Train image generator
    train_image_generator = image.ImageDataGenerator(**data_generator_args)
    train_mask_generator = image.ImageDataGenerator(**data_generator_args)

    train_image_generator.fit(X_train, seed=seed)
    train_mask_generator.fit(y_train, seed=seed)

    train_image_generator = train_image_generator.flow(X_train, batch_size=batch_size, seed=seed)
    train_mask_generator = train_mask_generator.flow(y_train, batch_size=batch_size, seed=seed)

    train_generator = zip(train_image_generator, train_mask_generator)

    return train_generator
