
import os
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import re
import json
import sys 

def image_to_keep(annotation_img,label_config):
    """
    Checks if an image has to be kept from the dataset based on 4 criteria:
        1) It contains a person (needs to be removed for ethical reasons)
        2) It contains less than 5 percent of 'indoor' pixels (otherwise the image is useless in the context of this study)
        3) It contains less than 6% of 'unlabelled' pixels
        4) The image is bigger than 526*256

    Args:
        annotation_img (numpy.ndarray): The annotated images
        label_config (String): the path to the labels configuration
        
    Returns a boolean: True if the image has to be kept, False otherwise
    """
    #these 2 lists will contain the labels names of each element in the "indoor" and "outdoor" categories
    indoor_labels,outdoor_labels = [],[]
    to_remove_label = 0 #person label

    try :
        #opening the label json file
        with open(label_config, 'r') as f:
            labels = json.load(f)
            #collecting the lebel numbers for the 'indoor' and 'outdoor' categories
            for category, labels_list in labels["categories"].items():
                if re.search(r'indoor', category):
                    indoor_labels.extend(labels_list)
                elif re.search(r'outdoor', category):
                    outdoor_labels.extend(labels_list)
                                
            annotations, counts = np.unique(annotation_img, return_counts=True)
            
            total_pixels = np.sum(counts)
            count_indoor_pixel = 0 #initialisation of the counts
            count_unlabelled_pixel = 0
            
            #calculating the percentage of 'indoor' and 'unlabelled' pixels
            for label,count in zip(annotations, counts):
                if label in indoor_labels :
                    count_indoor_pixel += count
                if label ==255:
                    count_unlabelled_pixel+= count
            if total_pixels!=0:
                percentage_indoor = count_indoor_pixel / total_pixels *100
                percentage_unlabelled = count_unlabelled_pixel / total_pixels *100
            else:
                percentage_indoor = 0
                percentage_unlabelled = 0
                
            #image size
            height, width = annotation_img.shape  

            
            if (to_remove_label in annotations) or (percentage_indoor >5) or (height<256) or (width<256) or (percentage_unlabelled>6) :
                #if the image contains a person, it needs to be removed
                return False
            else:
                return True
        
            
    except FileNotFoundError:
        print(f"Folder {annotations} not found.")
        sys.exit(1)
    #inspired from geeksforgeeks.org/json-parsing-errors-in-python/
    except json.JSONDecodeError:
        print(f"Error: The file {label_config} is not a valid JSON.")
        sys.exit(1)

def filter_data(annotations_directory,images_directory,label_config,filtered_images_directory,filtered_annotations_directory):
    """
    Filters images and annotations and copies the selected images into a filtered data directory.

    Args :
    annotations_directory (str): The path to the directory containing the annotation files.
    images_directory (str):  The path to the directory containing the image files.
    label_config (str) : The path to the label configuration.
    filtered_images_directory,filtered_annotations_directory (str): The paths to the directories where the filtered data must be copied to


    """
    count_kept_images =0 #to know how many images are kept after filtering
    
    #folders containing the filtered data
    os.makedirs(filtered_images_directory, exist_ok=True)
    os.makedirs(filtered_annotations_directory, exist_ok=True)
    
    #for each image og the dataset
    for image_name in tqdm(os.listdir(images_directory), desc="Copying good images in the folder filtered_data"):
        image_path = os.path.join(images_directory,image_name)
        #warning : in the dataset, the images are in a .jpeg format, but the annotations are in a .png format
        image_no_extension, _ = os.path.splitext(image_name)
        annotations_path = os.path.join(annotations_directory, image_no_extension+".png")

        try:
            annotation_img = cv2.imread(annotations_path, cv2.IMREAD_GRAYSCALE)
        
            if  image_to_keep(annotation_img,label_config) and os.path.exists(annotations_path):
                count_kept_images+=1
                #copying the images in a new folder
                shutil.copy(annotations_path, filtered_annotations_directory)
                shutil.copy(image_path, filtered_images_directory)
                
        except FileNotFoundError:
            print(f"File not found: {annotations_path}")
            return False
        except Exception:
            print(f"An error occurred while processing {annotations_path}")
            return False

    print (f"Number of images kept after filtering : {count_kept_images}")
            

if __name__ == '__main__':
    
    current_directory = os.getcwd()
    raw_annotations_directory = os.path.join(current_directory,'data', 'raw_data', 'annotations')
    raw_images_directory = os.path.join(current_directory,'data', 'raw_data', 'images')
    label_config = os.path.join(current_directory,'data','label_config.json')
    
    filtered_images_directory =  os.path.join(current_directory,'data', "filtered_data","filtered_images")
    filtered_annotations_directory =  os.path.join(current_directory,'data', "filtered_data","filtered_annotations")
  
    filter_data(raw_annotations_directory,raw_images_directory,label_config,filtered_images_directory,filtered_annotations_directory)


    