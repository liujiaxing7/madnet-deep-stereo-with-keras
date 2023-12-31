import os

import numpy as np
import tensorflow as tf
import argparse
from madnet import MADNet, colorize_img
from preprocessing import StereoDatasetCreator
from losses_and_metrics import SSIMLoss, EndPointError, Bad3
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='Script for inferencing MADNet')
parser.add_argument("--left_dir", help='Path to left images folder', required=True)
parser.add_argument("--right_dir", help='Path to right images folder', required=True)
parser.add_argument("--disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--num_adapt", help='Number of modules to adapt, needs to be an integer from 0-6',
                    default=0, type=int, required=False)
parser.add_argument("--mad_mode", help='Module selection method for MAD adaptation. \n'
                                       'Options are "random" and "sequential".',
                    default="random", required=False)
parser.add_argument("--search_range", help='Maximum dispacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_path",
                    help='Path for saving the adapted model (only needed if performing MAD)',
                    default=None, required=False)
parser.add_argument("--weights_path",
                    help='One of the following pretrained weights (will download automatically): '
                         '"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"'
                         'or a path to pretrained MADNet weights file (for fine turning)',
                    default=None, required=False)
parser.add_argument("--lr", help="learning rate (only needed if performing adaptation while inferencing.",
                    default=0.0001, type=float, required=False)
parser.add_argument("--height", help='Model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='Model image input height resolution', type=int, default=640)
parser.add_argument("--batch_size", help='Batch size to use during training', type=int, default=1)
parser.add_argument("--show_pred", help='Displays the models predictions', action="store_true", default=False)
parser.add_argument("--steps", help='Number of steps to inference, set to None to inference all the data',
                    default=None, type=int, required=False)
parser.add_argument('--bf', type=float, default=14.2)

args = parser.parse_args()

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def ScaleDepth(depth, bits=1):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if (depth_max - depth_min) > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        out = out.astype("uint8")
    elif bits == 2:
        out = out.astype("uint16")

    return out

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)

    return depth_img_rgb.astype(np.uint8)

def WriteDepth(depth, limg, path, name, bf):
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    name_png = os.path.splitext(name)[0] + ".png"
    output_gray = os.path.join(path, "gray", name_png)
    output_depth = os.path.join(path, "depth", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    MkdirSimple(output_gray)

    predict_np = tf.squeeze(depth)
    predict_np = predict_np.numpy()

    depth_img_float = predict_np  #bf / predict_np * 100  # to cm

    # depth_img = ScaleDepth(depth_img_float, bits=2)
    # cv2.imwrite(output_gray, depth_img)
    # return

    depth_img_int8 = ScaleDepth(depth_img_float, bits=1)

    color_img = cv2.applyColorMap(depth_img_int8, cv2.COLORMAP_HOT)
    limg_cv = limg  # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(depth_img_int8)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_depth)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_depth, depth_img_rgb)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)
    cv2.imwrite(output_gray, depth_img_float)

def getAbsdiff(depth_map, disparity_map, path):

    depth_map = np.squeeze(depth_map)
    print("depth map : ", depth_map)

    disparity_map =np.squeeze(disparity_map)
    print("disparity map: ", disparity_map)

    mask = disparity_map > 0

    # 归一化数据范围
    # depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_norm = depth_map
    depth_map_norm_mask = depth_map * mask
    # disparity_map_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_norm = disparity_map * mask
    # print("depth_map_norm: ", depth_map_norm)

    # 计算深度图和视差图的差异
    # diff_map = cv2.absdiff(depth_map_norm_mask.astype("float32"), disparity_map_norm.astype("float32"))
    diff_map = np.abs(depth_map_norm_mask.astype("float32") - disparity_map_norm.astype("float32"))
    # diff_map = cv2.bitwise_and(diff_map, mask)

    # 计算深度图和视差图的差异
    fig, axes = plt.subplots(4, 1, figsize=(6, 12))
    # axes[0].subplot(1, 3, 1)
    axes[0].imshow(depth_map_norm, cmap='gray')
    axes[0].set_title('Depth Map')
    axes[0].axis('off')
    cv2.imwrite(os.path.join(path, "depth_map_norm.png"), depth_map_norm)

    axes[1].imshow(depth_map_norm_mask, cmap='gray')
    axes[1].set_title('Depth Map Mask')
    axes[1].axis('off')
    cv2.imwrite(os.path.join(path, "depth_map_norm_mask.png"), depth_map_norm_mask)

    # axes[1].subplot(1, 3, 2)
    axes[2].imshow(disparity_map_norm, cmap='gray')
    axes[2].set_title('Disparity Map')
    axes[2].axis('off')
    cv2.imwrite(os.path.join(path, "disparity_map_norm.png"), disparity_map_norm)

    # axes[2].subplot(1, 3, 3)
    cax = axes[3].imshow(diff_map, cmap='jet')
    axes[3].set_title('Difference Map')
    # axes[2].colorbar()
    axes[3].axis('off')
    diff_map = cv2.applyColorMap(np.asarray(diff_map).astype("uint8"), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(path, "diff_map.png"), diff_map)

    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.05)
    plt.colorbar(cax, ax=axes[3])

    # 调整图像尺寸
    fig.set_size_inches(10, 15)

    plt.tight_layout()
    plt.show()
    plt.savefig("myplot.png")

def main(args):
    if args.output_path is None and args.num_adapt != 0:
        raise ValueError("No output_path provided for adaptation."
                         "Either set num_adapt=0 or provide a path to output_path"
                         f"Provided args: output_path: {args.output_path}, num_adapt: {args.num_adapt}")

    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        weights=args.weights_path,
        num_adapt_modules=args.num_adapt,
        mad_mode=args.mad_mode,
        search_range=args.search_range
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=SSIMLoss(),
        metrics=None,
        run_eagerly=False
    )
    # Get inferencing data
    predict_dataset = StereoDatasetCreator(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        shuffle=False,
        disp_dir=args.disp_dir
    )
    predict_ds = predict_dataset()
    # inference the dataset
    disparities = model.predict(predict_ds, steps=args.steps)
    # from keras.engine import data_adapter
    # inputs, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(predict_ds)

    # loss = self.compiled_loss(gt, final_disparity, sample_weight, regularization_losses=self.losses)

    # View disparity predictions
    filenames = predict_dataset.left_names_numpy
    if args.show_pred:
        for i in range(disparities.shape[0]):
            plt.axis("off")
            # plt.grid(visible=None)
            disp = tf.expand_dims(disparities[i, :, :, :], axis=0)
            plt.imshow(colorize_img(disp, cmap='jet')[0])
            plt.show()
            # filename = str(filenames[i].numpy())
            filename = args.left_dir + "/" + filenames[i]
            img = cv2.imread(filename)
            img = cv2.resize(img, (args.width, args.height))

            disp = tf.transpose(disp, perm=[0,3,1,2])
            WriteDepth(disp, img, args.output_path, filenames[i], args.bf)

            if args.disp_dir is not None:
                print(args.disp_dir + "/" + predict_dataset.disp_names_numpy[i])
                gt = predict_dataset._get_disp(predict_dataset.disp_names_numpy[i])
                print("max pixel: ", np.max(gt.numpy()))
                epe = EndPointError()
                epe.update_state(gt, disparities)
                print("EPE: ", epe.result())

                bad3 = Bad3()
                bad3.update_state(gt, disparities)
                print("Bad3: ", bad3.result())

                # getAbsdiff(disparities, predict_dataset.disp_names_numpy[0])
                getAbsdiff(disparities, gt, args.output_path, filenames[i])

    # save the checkpoint and saved_model if it was updated
    if args.num_adapt != 0:
        model.save_weights(args.output_path)


if __name__ == "__main__":
    main(args)