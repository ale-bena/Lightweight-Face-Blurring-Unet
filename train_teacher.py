import os
import argparse
import tensorflow as tf
from model_teacher import build_blur_unet
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import datetime

DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

@register_keras_serializable()
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

@register_keras_serializable()
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def process_path(img_path, tgt_path, img_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) # needs changing if png format is used for the dataset
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    tgt = tf.io.read_file(tgt_path)
    tgt = tf.image.decode_jpeg(tgt, channels=3)
    tgt = tf.image.resize(tgt, img_size)
    tgt = tf.cast(tgt, tf.float32) / 255.0

    return img, tgt

def load_dataset(image_dir, target_dir, img_size, batch_size, max_images=None, cache_path=None):
    # Images are loaded in an order to be processed in pairs
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]) # If using png format for the dataset change to .png
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".jpg")])

    if max_images is not None:
        image_files = image_files[:max_images]
        target_files = target_files[:max_images]

    image_ds = tf.data.Dataset.from_tensor_slices(image_files)
    target_ds = tf.data.Dataset.from_tensor_slices(target_files)

    dataset = tf.data.Dataset.zip((image_ds, target_ds))
    dataset = dataset.map(
        lambda x, y: process_path(x, y, img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=True
    )
    if cache_path:
        print(f"Caching dataset to {cache_path}")
        dataset = dataset.cache(cache_path)
    else:
        print("Caching dataset in memory")
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def train_model(resume_training, model_path, epochs, img_size, batch_size,
                train_images_dir, train_targets_dir, val_images_dir, val_targets_dir,
                max_train_images, max_val_images, output_dir, best_model_name,
                final_model_name, csv_log_name):
    
    if resume_training:
        print(f"Loading existing model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating new model...")
        model = build_blur_unet(input_shape=(*img_size, 3))

    model.compile(optimizer='adam', loss='mse', metrics=['mae', ssim_metric, psnr_metric])

    print("Loading datasets...")
    train_dataset = load_dataset(train_images_dir, train_targets_dir, img_size, batch_size,
                                 max_images=max_train_images,
                                 cache_path="/content/train_cache.tfdata")

    validation_dataset = load_dataset(val_images_dir, val_targets_dir, img_size, batch_size,
                                      max_images=max_val_images,
                                      cache_path="/content/val_cache.tfdata")

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, best_model_name)
    final_model_path = os.path.join(output_dir, final_model_name)
    csv_log_path = os.path.join(output_dir, csv_log_name)

    # TensorBoard setup
    log_dir = os.path.join(output_dir, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,     
        write_images=False,  
        update_freq='epoch'
    )

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path, save_best_only=True, monitor="val_loss", mode="min"
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, mode="min", start_from_epoch=20
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=6, min_lr=1e-6, verbose=1, mode="min", start_from_epoch=12
    )
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, append=True)

    print("Starting Teacher training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger, tensorboard_cb]
    )

    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    best_val_loss = min(history.history['val_loss'])
    print(f"Best model (lowest val_loss) at epoch {best_epoch} with val_loss = {best_val_loss:.4f}")

    if earlystop_cb.stopped_epoch > 0:
        print("Early stopping, training ended")
    else:
        print("Training completed")

    model.save(final_model_path)
    # model.save_weights(os.path.join(output_dir, "teacher.weights.h5"))
    print(f"Teacher model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")
    print(f"Training log saved to {csv_log_path}")


def main(args):
    print(f"Starting Teacher training with {args.epochs} epochs")

    # GPU configuration(works on Colab, not guaranteed to work in other situations)
    if args.gpu_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(e)

    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    for d in devices:
        if d.device_type == 'GPU':
            print(f"GPU: {d.name}, memory limit: {d.memory_limit / (1024 ** 3):.2f} GB")

    train_model(
        resume_training=args.resume_training,
        model_path=args.model_path,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_images_dir=args.train_images_dir,
        train_targets_dir=args.train_targets_dir,
        val_images_dir=args.val_images_dir,
        val_targets_dir=args.val_targets_dir,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        output_dir=args.output_dir,
        best_model_name=args.best_model_name,
        final_model_name=args.final_model_name,
        csv_log_name=args.csv_log_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Teacher Model")

    parser.add_argument("--resume_training", action="store_true",
                        help="Resume training from existing model")
    parser.add_argument("--model_path", type=str, default="models_teacher/teacher.keras",
                        help="Path to existing model or where to save new model")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs")

    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128],
                        help="Input image size (height width)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")

    # Dataset paths
    parser.add_argument("--train_images_dir", type=str, default="./dataset1/train",
                        help="Path to training images directory")
    parser.add_argument("--train_targets_dir", type=str, default="./dataset1/train_blur",
                        help="Path to training targets (blurred) directory")
    parser.add_argument("--val_images_dir", type=str, default="./dataset1/val",
                        help="Path to validation images directory")
    parser.add_argument("--val_targets_dir", type=str, default="./dataset1/val_blur",
                        help="Path to validation targets (blurred) directory")

    parser.add_argument("--max_train_images", type=int, default=9600,
                        help="Maximum number of training images")
    parser.add_argument("--max_val_images", type=int, default=2400,
                        help="Maximum number of validation images")

    parser.add_argument("--gpu_growth", action="store_true",
                        help="Enable GPU memory growth")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="models_teacher",
                        help="Output directory for saving models and logs")
    parser.add_argument("--best_model_name", type=str, default="best_teacher.keras",
                        help="Name for the best model checkpoint")
    parser.add_argument("--final_model_name", type=str, default="teacher_final.keras",
                        help="Name for the final model")
    parser.add_argument("--csv_log_name", type=str, default="teacher_training_log.csv",
                        help="Name for the CSV training log file")

    args = parser.parse_args()
    main(args)


