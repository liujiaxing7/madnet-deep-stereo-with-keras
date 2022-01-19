import os
import tensorflow as tf
import numpy as np


class StereoDatasetCreator():
    """
    Takes paths to left and right stereo image directories
    and creates a dataset that returns a batch of left 
    and right images, (Optional) returns the disparities as a target
    using the disparities directories.
    Init Args:
        left_dir: path to left images folder
        right_dir: path to right images folder
        batch_size: desired batch size 
        height: desired height of the image (will be reshaped to this height if necessary)
        width: desired width of the image (will be reshaped to this width if necessary)
        shuffle: True/False
        (Optional) disp_dir: path to disparity maps folder
    Returns:
        object that can be called to return a tf.data.Dataset
        dataset will return values of the form: 
            {'left_input': (batch, height, width, 3), 'right_input': (batch, height, width, 3)}, (Optional) (batch, height, width, 1) else None

    This can prepare MADNet data for training/evaluation and prediction
    """
    def __init__(self, left_dir, right_dir, height, width, batch_size=1, shuffle=False, disp_dir=None):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle

        self.left_names = tf.constant(sorted([name for name in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{name}")]))
        self.right_names = tf.constant(sorted([name for name in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{name}")]))
        if self.disp_dir is not None:
            self.disp_names = tf.constant(sorted([name for name in os.listdir(disp_dir) if os.path.isfile(f"{self.disp_dir}/{name}")]))

        # Check that there is a left image for every right image
        self.num_left = len(self.left_names)
        self.num_right = len(self.right_names)
        if self.num_left != self.num_right:
            raise ValueError(f"Number of right and left images do now match. Left number: {self.num_left}. Right number: {self.num_right}")

    def _get_image(self, path):
        """
        Get a single image helper function
        Converts image to float32, normalises values to 0-1
        and resizes to the desired shape
        Args:
            path to image (will be in Tensor format, since its called in a graph)
        Return:
            Tensor in the shape (height, width, 3)
        """
        # Using tf.io.read_file since it can take a tensor as input
        raw = tf.io.read_file(path)
        # Converts to float32 and normalises values
        image = tf.io.decode_image(raw, channels=3, dtype=tf.float32, expand_animations=False)
        # Change dimensions to the desired model dimensions
        image = tf.image.resize(image, [self.height, self.width], method="bilinear")
        return image

    def readPFM(self, file):
        """
        Load a pfm file as a numpy array
        Args:
            file: path to the file to be loaded
        Returns:
            content of the file as a numpy array
            with shape (height, width, channels)
        """
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dims = file.readline()
        try:
            width, height = list(map(int, dims.split()))
        except:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    def _get_pfm(self, path):
        """
        Reads a single pfm disparity file and
        returns a numpy disparity map
        Args:
            path: path to the disparity file (will be in Tensor format, since its called in a graph)
        Returns:
            Tensor disparity map with shape (height, width, 1) 
        """
        # Convert tensor to a string
        path = path.numpy().decode("ascii")

        #disp_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        disp_map = self.readPFM(path)

        # Set inf values to 0 (0 is infinitely far away, so basically the same)
        disp_map[disp_map==np.inf] = 0
        # convert values to positive
        if disp_map.mean() < 0:
            disp_map *= -1
        # Change dimensions to the desired (height, width, channels)
        #disp_map = tf.expand_dims(disp_map, axis=-1)
        disp_map = tf.image.resize(disp_map, [self.height, self.width], method="bilinear")
        return disp_map

    def _process_single_batch(self, index):
        """
        Processes a single batch using index to find the files
        Args: 
            index: Tensor integer
        Returns:
            stereo input dictionary, target
        """
        left_name = self.left_names[index]
        right_name = self.right_names[index]
        left_image = self._get_image(f"{self.left_dir}/" +  left_name)
        right_image = self._get_image(f"{self.right_dir}/" + right_name)

        disp_map = None  
        if self.disp_dir is not None:
            disp_name = self.disp_names[index]  
            # wrapping in py_function so that the function can execute eagerly and run non tensor ops
            disp_map = tf.py_function(func=self._get_pfm, inp=[f"{self.disp_dir}/" + disp_name], Tout=tf.float32)

        return {'left_input': left_image, 'right_input': right_image}, disp_map

    def __call__(self):
        """
        Creates and returns a tensorflow data.Dataset
        The dataset is shuffled, batched and prefetched
        """
        indexes = list(range(self.num_left))   
        indexes_ds = tf.data.Dataset.from_tensor_slices(indexes)
        if self.shuffle == True:
            indexes_ds.shuffle(buffer_size=self.num_left, seed=101, reshuffle_each_iteration=False)

        ds = indexes_ds.map(self._process_single_batch)
        ds = ds.batch(batch_size=self.batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=10)
        return ds


class StereoGenerator(tf.keras.utils.Sequence):
    """
    This method is currently not working.
    Please use the StereoDatasetCreator instead for datapreperation.
    The Input data has shape (None, None, None, None) for each image when training
    Takes paths to left and right stereo image directories
    and creates a generator that returns a batch of left 
    and right images.
    
    """
    def __init__(self, left_dir, right_dir, batch_size, height, width, shuffle):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle

        self.left_paths = [path for path in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{path}")]
        self.right_paths = [path for path in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{path}")]
        # Check that there is a left image for every right image
        self.num_left = len(self.left_paths)
        self.num_right = len(self.right_paths)
        if self.num_left != self.num_right:
            raise ValueError(f"Number of right and left images do now match. Left number: {self.num_left}. Right number: {self.num_right}")
        # Check if images names are identical
        self.left_paths.sort()
        self.right_paths.sort()
        if self.left_paths != self.right_paths:
            raise ValueError("Left and right image names do not match. Please make sure left and right image names are identical")

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.num_left // self.batch_size


    def _get_image(self, image_dir, image_name):
        # get a single image helper function
        image = tf.keras.preprocessing.image.load_img(f"{image_dir}/{image_name}")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (self.height, self.width)).numpy()
        return image_arr/255.

    def __getitem__(self, batch_index):
        index = batch_index * self.batch_size
        left_batch = self.left_paths[index: self.batch_size + index]
        right_batch = self.right_paths[index: self.batch_size + index]

        left_images = tf.constant([self._get_image(self.left_dir, image_name) for image_name in left_batch])
        right_images = tf.constant([self._get_image(self.right_dir, image_name) for image_name in right_batch])
        return {'left_input': left_images, 'right_input': right_images}, None


