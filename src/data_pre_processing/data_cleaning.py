import os
from tqdm import tqdm
import numpy as np
import shutil
import random
import cv2
import tensorflow as tf
import math

def resize_img(image,size=(256,256)):
    """This function resizes an image. 256*256 if no size specified.

    Args:
        image : The image to be resized.
        size (tuple, optional): Defaults to (256,256).
        
    Returns: 
        numpy.ndarray : The image resized.
    """

    #it is important to use nearest neighbour method to avoid introducing new labels in semantic maps
    resized_img = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

    return resized_img

def normalisation(image):
    """This function sets the pixels of qn imqge between 0 and 1.
    Args:
        image : The image to be resized.
        
    Returns: 
        numpy.ndarray : The image normalised.

    """
    normalized_image = image.astype(np.float32)
    return normalized_image
    
def preprocess(img,annotation_img,size=(256,256)):
    """This function does the preprocessing for a couple of images: the image and its associated semantic map.

    Args:
        img (np.ndarray) : The image to preprocess.
        annotation_img (np.ndarray) : The associated semantic map.
        size (tuple, optional): The desired size of the images.
        
    Returns: 
        numpy.ndarray : The preprocessed image.
        None if the image and the semantic map are not of the same shapes (ie they do not match)

    """
    if (img.shape[:2] == annotation_img.shape):
        #resising image
        annotation, image = resize_img(annotation_img, (256,256)) , resize_img(img, (256,256))
        #normalising image
        normalised_imaged, normalised_annotation= image.astype(np.float32) , annotation.astype(np.uint8)
        return normalised_imaged, normalised_annotation
    else:
        return None


def split_train_val(images_directory, annotations_directory, validation_ratio = 0.2):
    """
    This function splits the dataset into a training and validation sets, with the ration validation_ratio.
    """
    
    
    all_images = os.listdir(images_directory)
    random.shuffle(all_images)
    nb_of_val_images = int(len(all_images) * validation_ratio)
    
    validation_images = all_images[:nb_of_val_images]
    train_images =  all_images[nb_of_val_images:]
    
    train_image_paths = [os.path.join(images_directory, image) for image in train_images]
    val_image_paths = [os.path.join(images_directory, image) for image in validation_images]

    train_annotation_paths = [os.path.join(annotations_directory, os.path.splitext(image)[0] + ".png") for image in train_images]
    val_annotation_paths = [os.path.join(annotations_directory, os.path.splitext(image)[0] + ".png") for image in validation_images]

    return train_image_paths, train_annotation_paths, val_image_paths, val_annotation_paths


    
def run_preprocessing(images_filtered,annotations_filtered,pre_processed_data_directory):
    """This function runs the preprocessing on a dataset.
    Args:
        images_filtered (str) : The path to the images folder.
        annotations_filtered (str) : The path to the annotations (semantic maps) folder.
        pre_processed_data_directory: The path where to preprocessed images must be saved. It is created in this function.
         

    """
    #creating the folder to save the images
    folder_clean_images = os.path.join(pre_processed_data_directory,'images')
    folder_clean_annotations = os.path.join(pre_processed_data_directory,'annotations')
    os.makedirs(folder_clean_images, exist_ok=True)
    os.makedirs(folder_clean_annotations, exist_ok=True)


    
    #for each image of the dataset
    for filtered_image_path, filtered_annotations_path in tqdm(zip(images_filtered, annotations_filtered), desc="Preprocessing data"):
        try:
            #opening the image and semantic map
            img = cv2.imread(filtered_image_path)
            annotation_img = cv2.imread(filtered_annotations_path, cv2.IMREAD_GRAYSCALE)
        
            clean_image, clean_annotation = preprocess(img,annotation_img)
            if clean_image is not None and clean_annotation is not None:
                
                clean_image_path = os.path.join(folder_clean_images,os.path.basename(filtered_image_path))
                clean_annotation_path  = os.path.join(folder_clean_annotations,os.path.basename(filtered_annotations_path))
                
                cv2.imwrite(clean_image_path, clean_image)
                cv2.imwrite(clean_annotation_path, clean_annotation)
                

        except FileNotFoundError:
            print(f"Image not found: {filtered_image_path} and {filtered_annotations_path}.")
        except Exception as e:
            print(f"An error occurred while processing {filtered_image_path} and {filtered_annotations_path}: {e}.")

if __name__ == '__main__':
    
    current_directory = os.getcwd()
    annotations_filtered = os.path.join(current_directory,'data', 'filtered_data', 'filtered_annotations')
    images_filtered = os.path.join(current_directory,'data', 'filtered_data', 'filtered_images')

    pre_processed_data_directory = os.path.join( current_directory,'data', 'test')
    
    
    train_image_paths, train_annotation_paths, val_image_paths, val_annotation_paths = split_train_val(images_filtered, annotations_filtered)
    
    train_directory = os.path.join(pre_processed_data_directory,'train')
    val_directory = os.path.join(pre_processed_data_directory,'val')

    run_preprocessing(val_image_paths,val_annotation_paths,val_directory)
    run_preprocessing(train_image_paths,train_annotation_paths,train_directory)

    

