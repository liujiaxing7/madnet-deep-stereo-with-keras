import os
import tensorflow as tf
import argparse
from subclassed_madnet import MADNet
from preprocessing import StereoDatasetCreator
from losses_and_metrics import Bad3, EndPointError, ReconstructionLoss


print("\nTensorFlow Version: {}".format(tf.__version__))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


parser=argparse.ArgumentParser(description='Script for training subclassed MADNet')
parser.add_argument("--train_left_dir", help='path to left images folder', required=True)
parser.add_argument("--train_right_dir", help='path to right images folder', required=True)
parser.add_argument("--train_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--val_left_dir", help='path to left images folder', default=None, required=False)
parser.add_argument("--val_right_dir", help='path to right images folder', default=None, required=False)
parser.add_argument("--val_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--shuffle", help='shuffle training dataset', action="store_true", default=False)
parser.add_argument("--search_range", help='maximum dispacement (ie. smallest disparity)', default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir", help='path to folder for outputting tensorboard logs and saving model weights', required=True)
parser.add_argument("--checkpoint_path", help="path to pretrained MADNet checkpoint file (for fine turning)", default=None, required=False)
parser.add_argument("--lr", help="initial value for learning rate",default=0.0001, type=float, required=False)
parser.add_argument("--height", help='model image input height resolution', type=int, default=320)
parser.add_argument("--width", help='model image input height resolution', type=int, default=1216)
parser.add_argument("--batch_size", help='batch size to use during training',type=int,default=1)
parser.add_argument("--num_epochs", help='number of training epochs', type=int, default=1000)
parser.add_argument("--epoch_steps", help='training steps per epoch', type=int, default=1000)
parser.add_argument("--save_freq", help='model saving frequncy per steps', type=int, default=1000)
parser.add_argument("--use_full_res_loss", help='for using only the final resolution loss during backpropagation', action="store_true", default=False)
parser.add_argument("--epoch_evals", help='number of epochs per evaluation', type=int, default=1)
parser.add_argument("--eval_steps", help='number of batches to process per evaluation', type=int, default=1)
args=parser.parse_args()


def main(args):
    train_left_dir = args.train_left_dir
    train_right_dir = args.train_right_dir
    train_disp_dir = args.train_disp_dir
    val_left_dir = args.val_left_dir
    val_right_dir = args.val_right_dir
    val_disp_dir = args.val_disp_dir
    shuffle = args.shuffle
    search_range = args.search_range
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    lr = args.lr
    height = args.height
    width = args.width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    epoch_steps = args.epoch_steps
    save_freq = args.save_freq
    use_full_res_loss = args.use_full_res_loss
    epoch_evals = args.epoch_evals
    eval_steps = args.eval_steps

    run_eager = True

    # Create output folder if it doesnt already exist
    os.makedirs(output_dir, exist_ok=True)

    with strategy.scope():
        # Initialise the model
        model = MADNet(height=height, width=width, search_range=search_range, batch_size=batch_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if checkpoint_path is not None:
            model.load_weights(checkpoint_path)

        model.compile(
            optimizer=optimizer, 
            # losses and metrics below are only needed for evaluation
            loss=[ReconstructionLoss()],
            metrics=[EndPointError(), Bad3()],
            run_eagerly = run_eager  
        )

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
    train_ds = train_dataset().repeat()
    # Get validation data
    val_ds = None
    if val_left_dir is not None and val_right_dir is not None and val_disp_dir is not None:
        val_dataset = StereoDatasetCreator(
            left_dir=val_left_dir, 
            right_dir=val_right_dir, 
            batch_size=batch_size, 
            height=height, 
            width=width,
            shuffle=shuffle,
            disp_dir=val_disp_dir
            ) 
        val_ds = val_dataset().repeat()
    
    # Create callbacks
    def scheduler(epoch, lr):
        if epoch < 400000:
            lr = lr
        elif epoch < 600000:
            lr = lr * 0.5
        else:
            # learning_rate * decay_rate ^ (global_step / decay_steps)
            decay_rate = 0.5       
            lr = lr * 0.5 * decay_rate ** (epoch // 600000)
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr

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


    model.use_full_res_loss = use_full_res_loss
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
        ],
        validation_data=val_ds,
        validation_steps=eval_steps,
        validation_freq=epoch_evals # number epoch evaluations 
    )

if __name__ == "__main__":
    main(args)