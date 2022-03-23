import tensorflow as tf                                                                                                                                                                                           
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import cv2
import time
import datetime
import sys

#from qarpo.demoutils import *

import logging as log
import argparse

print("TensorFlow version = {}".format(tf.__version__))
#print("Does TensorFlow have the Intel optimizations: {}".format(tf.python._pywrap_util_port.IsMklEnabled()))
print('Imported Python modules successfully.')

def set_OMP_vars(num_cores):
    # export TF_DISABLE_MKL=1
    os.environ["TF_DISABLE_MKL"]  = "0"  # Disable Intel optimizations

    # export MKLDNN_VERBOSE=1
    #os.environ["MKLDNN_VERBOSE"]  = "1"     # 1 = Print log statements; 0 = silent

    os.environ["OMP_NUM_THREADS"] = num_cores   # Number of physical cores
    os.environ["KMP_BLOCKTIME"]   = "1"

    # If hyperthreading is enabled, then use
    os.environ["KMP_AFFINITY"]    = "granularity=thread,compact,1,0"

    # If hyperthreading is NOT enabled, then use
    #os.environ["KMP_AFFINITY"]   = "granularity=thread,compact"

def load_input_data(data_directory):
    print("data directory is: ", data_directory)
    (ds), ds_info =  tfds.load(data_directory
                               , data_dir="."
                               , shuffle_files=True, split='train'
                               , with_info=True
                               , as_supervised=True)

    assert isinstance(ds, tf.data.Dataset)
    print(ds_info)
    return(ds, ds_info)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def augment_img(image, label):
    """Augment images: `uint8` -> `float32`."""

    image = tf.image.random_flip_left_right(image) # Random flip Left/Right
    image = tf.image.random_flip_up_down(image)    # Random flip Up/Down

    return tf.cast(image, tf.float32) / 255., label # Normalize 0 to 1 for pixel values

def save_test_data(ds_test, test_data_dir):
    images = []
    labels = []
    for ex_img, ex_label in ds_test:
        images.append(ex_img.numpy())
        labels.append(ex_label.numpy())
        
    np.savez(os.path.join(test_data_dir, "testdata.npz"), np.asarray(images), np.asarray(labels))

def preprocessing(ds, ds_info, output_dir, test_data_dir):
    n = ds_info.splits['train'].num_examples
    train_split_percentage = 0.80
    train_batch_size = 128
    test_batch_size = 16
  
    # Get train dataset
    ds_train = ds.take(int(n * train_split_percentage))
    ds_train = ds_train.map(augment_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(int(n * train_split_percentage))
    ds_train = ds_train.batch(train_batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Get test dataset
    ds_test = ds.skip(int(n * train_split_percentage)).map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(test_batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    
    save_test_data(ds_test, test_data_dir)
    
    return(ds_train, ds_test)

def create_model(ds_info):
    inputs = tf.keras.layers.Input(shape=ds_info.features['image'].shape)
    conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu")(conv)
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

    conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv)
    concat = tf.keras.layers.concatenate([maxpool, conv])
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

    conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
    conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv)
    concat = tf.keras.layers.concatenate([maxpool, conv])
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

    conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
    conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv)
    concat = tf.keras.layers.concatenate([maxpool, conv])
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

    flat = tf.keras.layers.Flatten()(maxpool)
    dense = tf.keras.layers.Dense(128)(flat)
    drop = tf.keras.layers.Dropout(0.5)(dense)

    predict = tf.keras.layers.Dense(ds_info.features['label'].num_classes)(drop)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[predict])

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
                  , optimizer='adam'
                  , metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    model.summary()
    return(model)
    
def create_callbacks(output_dir):
    # Create a callback that saves the model
    model_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir,
                                                         save_best_only=True,
                                                         verbose=1)

    # Create callback for Early Stopping of training
    # Stop once validation loss plateaus for patience epochs   
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=8) 

    # TensorBoard logs
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    return(checkpoint_callback, early_stopping_callback, tb_callback)


def train_model(model, epochs, ds_train, ds_test, checkpoint_callback, early_stopping_callback, tb_callback):
    history = model.fit(ds_train
                        , epochs=epochs
                        , validation_data=ds_test
                        , callbacks=[checkpoint_callback, early_stopping_callback, tb_callback]
    )
    return history
    
def main():
    # Set up logging
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False,
                    default='colorectal_histology', help="Path to input")
    ap.add_argument("-o", "--output", required=False,
                    default='results/', help="Output directory")
    ap.add_argument("-e", "--epochs", required=False,
                    default=5, help="Number of epochs")
    ap.add_argument("-d", "--device", required=False,
                    default="CPU", help="Targeted device")
    ap.add_argument("-c", "--cores", required=False,
                    default=40, help="Number of cores")
    ap.add_argument("-s", "--test_data_dir", required=False,
                    default="./test_data", help="Directory to save processed test data for processing")
  
    args = vars(ap.parse_args())
    
    if not args['input']:
            raise Exception("Specify input collateral")
    
    data_directory = args['input']
    assert(os.path.isdir(data_directory), "Input data directory {0} does not exist!".format(data_directory))
    output_dir = args['output']
    assert(os.path.isdir(output_dir), "Output directory {0} does not exist!".format(output_dir))
    epochs = args['epochs']
    device = args["device"]
    num_cores = args["cores"]
    test_data_dir = args["test_data_dir"]
    assert(os.path.isdir(test_data_dir), "Test data directory {0} does not exist!".format(test_data_dir))

    if device == "CPU":
        log.info("Set OMP parameters")
        set_OMP_vars(num_cores)
    
    log.info("Getting data")
    ds, ds_info = load_input_data(data_directory)
    
    log.info("Preprocessing data")
    ds_train, ds_test = preprocessing(ds, ds_info, output_dir, test_data_dir)
    
    log.info("Create callbacks")
    checkpoint_callback, early_stopping_callback, tb_callback = create_callbacks(output_dir)
    
    log.info("create model")
    model = create_model(ds_info)
  
    log.info("Start training")
    #job_id = os.environ['PBS_JOBID']   
    history = train_model(model, int(epochs), ds_train, ds_test, checkpoint_callback, early_stopping_callback, tb_callback)
      # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
                 tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60) 
    print("Frozen model layers: ")
    for layer in layers:
            print(layer)
    print("-" * 60) 
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=output_dir,
                      name="frozen_histology.pb",
                      as_text=False)
    
    log.info("End training")
    
if __name__ == '__main__':
    print("Start training")
    sys.exit(main() or 0)
