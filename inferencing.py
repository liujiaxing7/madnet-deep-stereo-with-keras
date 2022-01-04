import os
import tensorflow as tf
import argparse
from custom_models import *
from preprocessing import StereoDatasetCreator
import matplotlib.pyplot as plt
from datetime import datetime

print("\nTensorFlow Version: {}".format(tf.__version__))


parser=argparse.ArgumentParser(description='Script for training MADNet')
parser.add_argument("--left_dir", help='path to left images folder', required=True)
parser.add_argument("--right_dir", help='path to right images folder', required=True)
parser.add_argument("--mad_pred", help='use modular adaptation while inferencing', action="store_true", default=False)
parser.add_argument("--num_adapt", help='number of modules to adapt', default=1, type=int, required=False)
parser.add_argument("--search_range", help='maximum dispacement (ie. smallest disparity)', default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir", help='path to folder for saving updated model (only needed if performing MAD)', default=None, required=False)
parser.add_argument("--checkpoint_path", help="path to pretrained MADNet checkpoint file", required=True)
parser.add_argument("--lr", help="learning rate (only used if weights are a checkpoint)", default=0.0001, type=float, required=False)
parser.add_argument("--height", help='model image input height resolution', type=int, default=320)
parser.add_argument("--width", help='model image input height resolution', type=int, default=1216)
parser.add_argument("--batch_size", help='batch size to use during training',type=int, default=1)
parser.add_argument("--show_pred", help='displays the models predictions', action="store_true", default=False)
parser.add_argument("--steps", help='number of steps to inference, set to None to inference all the data', default=None, type=int, required=False)

args=parser.parse_args()



def main(args):
    left_dir = args.left_dir
    right_dir = args.right_dir
    mad_pred = args.mad_pred
    num_adapt = args.num_adapt
    search_range = args.search_range
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    lr = args.lr
    height = args.height
    width = args.width
    batch_size = args.batch_size
    show_pred = args.show_pred
    steps = args.steps

    run_eager = True


    # Initialise the model
    model = MADNet(height=height, width=width, search_range=search_range, batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, 
        run_eagerly=run_eager   
    )
    model.load_weights(checkpoint_path).expect_partial()


    # Get training data
    predict_dataset = StereoDatasetCreator(
        left_dir=left_dir, 
        right_dir=right_dir, 
        batch_size=batch_size, 
        height=height, 
        width=width
        ) 

    predict_ds = predict_dataset()
    # set model attributes
    model.run_eagerly = run_eager
    model.MAD_predict = mad_pred
    model.num_adapt_modules = num_adapt
    # inference the dataset
    disparities = model.predict(predict_ds, steps=steps)
    # View disparity predictions
    if show_pred:
        for i in range(disparities.shape[0]):
            plt.axis("off")
            plt.grid(visible=None)
            plt.imshow(disparities[i])
            plt.show()

    # save the checkpoint and saved_model if it was updated
    if output_dir is not None and mad_pred == True:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.now()
        current_time = now.strftime("%Y%m%dT%H%M%SZ")
        model_path = f"{output_dir}/MADNet-{current_time}.ckpt"
        model.save_weights(model_path)


if __name__ == "__main__":
    main(args)
