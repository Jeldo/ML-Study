import os
import time
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
try:
    import data_util
except ImportError:
    from dataset import data_util


def load_image(im_fn, image_size):
    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1] # rgb converted\
    resize_scale = 1 / 2
    h, w, _ = high_image.shape

    h_edge = h - image_size
    w_edge = w - image_size

    h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
    w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
    
    high_image = high_image[h_start:h_start+image_size, w_start:w_start+image_size, :]
    low_image = cv2.resize(high_image, (0, 0), fx=1, fy=1)
    #low_image = cv2.resize(low_image, (0, 0), fx=1/resize_scale, fy=1/resize_scale)
    
    return high_image,low_image

def get_record(image_path):
    images = glob.glob(image_path)
    print('%d files found' % (len(images)))
    if len(images) == 0:
        raise FileNotFoundError('check your training dataset path')
    index = list(range(len(images)))
    while True:
        random.shuffle(index)
        for i in index:
            im_fn = images[i]
            yield im_fn #high_image, low_image


def generator(image_path, image_size=512, batch_size=32):
    high_images = []
    low_images = []
    
    for im_fn in get_record(image_path):
        try:
            high_image,low_image = load_image(im_fn, image_size)
            high_images.append(high_image)
            low_images.append(low_image)
            if len(high_images) == batch_size:
                yield high_images,low_images
                high_images = []
                low_images = []

        except FileNotFoundError as e:
            print(e)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

def get_generator(image_path, **kwargs):
    return generator(image_path, **kwargs)


def get_batch(image_path, num_workers, **kwargs):
    try:
        generator = get_generator(image_path, **kwargs)
        enqueuer = data_util.GeneratorEnqueuer(generator, use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_ouptut = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
