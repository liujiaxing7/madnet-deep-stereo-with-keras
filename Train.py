import os
import tensorflow as tf
import argparse
from custom_models import MADNet
from preprocessing import StereoDatasetCreator


print("\nTensorFlow Version: {}".format(tf.__version__))



parser=argparse.ArgumentParser(description='Script for training MADNet')
parser.add_argument("--train_left_dir", help='path to left images folder', required=True)
parser.add_argument("--train_right_dir", help='path to right images folder', required=True)
parser.add_argument("--train_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--shuffle", help='shuffle training dataset', action="store_true", default=False)
parser.add_argument("--search_range", help='maximum dispacement (ie. smallest disparity)', default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir", help='path to folder for outputting tensorboard logs and saving model weights', required=True)
parser.add_argument("--weights_path", help="path to pretrained MADNet saved model (for fine turning)", default=None, required=False)
parser.add_argument("--lr", help="initial value for learning rate",default=0.0001, type=float, required=False)
parser.add_argument("--height", help='model image input height resolution', type=int, default=320)
parser.add_argument("--width", help='model image input height resolution', type=int, default=1216)
parser.add_argument("--batch_size", help='batch size to use during training',type=int,default=1)
parser.add_argument("--num_epochs", help='number of training epochs', type=int, default=1000)
parser.add_argument("--epoch_steps", help='training steps per epoch', type=int, default=1000)
parser.add_argument("--save_freq", help='model saving frequncy per steps', type=int, default=10)
args=parser.parse_args()


def main(args):
    train_left_dir = args.train_left_dir
    train_right_dir = args.train_right_dir
    train_disp_dir = args.train_disp_dir
    shuffle = args.shuffle
    search_range = args.search_range
    output_dir = args.output_dir
    weights_path = args.weights_path
    lr = args.lr
    height = args.height
    width = args.width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    epoch_steps = args.epoch_steps
    save_freq = args.save_freq

    run_eager = True
    if train_disp_dir is None:
        # dont need to train eagerly when not using gt disparities 
        run_eager = False

    # Create output folder if it doesnt already exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialise the model
    model = MADNet(height=height, width=width, search_range=search_range, batch_size=batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, 
        run_eagerly = run_eager  
    )

    if weights_path is not None:
        #model = tf.keras.models.load_model(weights_path, custom_objects={"MADNet": MADNet}, compile=True)
        #model = tf.saved_model.load(weights_path)
        model.load_weights(weights_path)


    # Get training data
    train_dataset = StereoDatasetCreator(
        left_dir=train_left_dir, 
        right_dir=train_right_dir, 
        batch_size=batch_size, 
        height=height, 
        width=width,
        shuffle=shuffle,
        disp_dir=train_disp_dir
        ) 

    train_ds = train_dataset()


    # Create callbacks
    def scheduler(epoch, lr):
        if epoch < 400000:
            return lr
        elif epoch < 600000:
            return lr * 0.5
        else:
            # learning_rate * decay_rate ^ (global_step / decay_steps)
            decay_rate = 0.5       
            return lr * 0.5 * decay_rate ** (epoch // 600000)

    schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + "/checkpoints/epoch-{epoch:04d}-MADNet.ckpt",
        save_freq=save_freq,
        save_weights_only=True,
        verbose=0
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=output_dir + "/tensorboard",
        histogram_freq=1,
        write_steps_per_second=True,
        update_freq="batch"
        )

    # Fit the model
    history = model.fit(
        x=train_ds,
        epochs=num_epochs,
        verbose=1,
        steps_per_epoch=epoch_steps,
        callbacks=[
            tensorboard_callback,
            save_callback,
            schedule_callback
        ]
    )

    model.save(f"{output_dir}/MADNet-{num_epochs}-epochs-{epoch_steps}-epoch_steps")

if __name__ == "__main__":
    main(args)