import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from src.data_pre_processing.visualisation import *
import json
import re



def images_size_exploration(annotation_img):  
    """  This function returns the size of an image.
 
    Parameters:
    annotation_img (numpy.ndarray): The image to analyse. It should be in a gray scale.
 
    Returns:
    tuple: The size of the image. Returns 0 if the image is empty
    """
    try:
        height, width = annotation_img.shape  
    except ValueError:
        height, width = 0,0
        print(f"Image annotation_img is empty.")

    return (width, height)
        

def pixel_label_exploration(annotation_img,labels_json,pixel_label_counts,image_label_counts,pixel_min,pixel_max):
    """  This function counts the number of occurrences of each label in an image and updates the provided dictionaries.
    2 dictionaries are updated : one for counting the number of pixels per label and one for 
    counting the number of images containing each label.
    
    For each label contained in the image, the corresponding count in image_label_counts will be 
    incremented by 1, and the number of pixels for that label will be added to pixel_label_counts
    
     
    Parameters:
    annotation_img (numpy.ndarray): The image to analyse. It should be in a gray scale, and the value of each pixel stands for a label.
    labels_json (dict): Dictionary containing the label configuration: 
                        The "label_names" key matches the pixel values with the label names. The "category" key indicates the labels belonging in each category.
    pixel_label_counts (dict): Dictionary to update. It stores the count of pixels for each label in each category.
    image_label_counts (dict): Dictionary to update. It stores the count of images containing each label in each category.
    pixel_min (float) : minimum value taken by a pixel. The goal of these min and max is to know the range of values taken by the pixels (excluding 255 as 255 is the "unlabelled" annotation)
    pixel_min (float) : minimum value taken by a pixel
    
    Returns:
    The two updated dictionaries.
    Warning: Pixel counts are not sorted. The order is the same than in the json file.
    """
    #counting the number of occurence for each label of the image
    unique_labels, counts = np.unique(annotation_img, return_counts=True)
    
    #checking in which category the labels belong to
    for label, count in zip(unique_labels, counts):
        pixel_min = min(pixel_min,label)
        if label != 255:
            pixel_max = max(pixel_max,label)
        for category, labels in labels_json["categories"].items():
            #if this label belongs to this category, then add it 
            if (label) in labels:
                label_name = labels_json["label_names"].get(str(label))
                pixel_label_counts[category][label_name] += count
                image_label_counts[category][label_name] += 1
    return pixel_label_counts,image_label_counts,pixel_min,pixel_max
    

def unlabelled_pixels_analysis(annotation_img):
    """  This function returns the percentage of 'unlabelled' pixels in an image
        255 is the value for an 'unlabeled' pixel.
        
    Parameters:
    annotation_img (numpy.ndarray): The image to analyse. It should be in a gray scale.
 
    Returns:
    float: percentage of 'unlabelled' pixels in the image. Returns None if the image is empty
    """
    try:
        height, width = annotation_img.shape
        nb_of_pixels_total = height * width

        unique_labels,counts = np.unique(annotation_img,return_counts=True)   
        nb_of_pixel_annotated = np.sum(counts)
        #check that all the pixels of the image are annotated
        if nb_of_pixel_annotated != nb_of_pixels_total:
            print(f'Warning: some pixels are not annotated in image {annotation_img}')
        #calculate the percentage of unlabelled pixels for each image
        percentage_unlabelled = 0
        for label,count in zip( unique_labels,counts):
            if label == 255:
                percentage_unlabelled = count/nb_of_pixels_total *100
                break 
    except ValueError:
        percentage_unlabelled = None
    return percentage_unlabelled

            
            
def calculate_indoor_pixel_percentage(annotation_img,labels_json,indoor_categories):
    """  This function returns the percentage of 'indoor' pixels in an image.
    
        
    Parameters:
    annotation_img (numpy.ndarray): The image to analyse. It should be in a gray scale.
    labels_json (dict): Dictionary containing the label configuration: 
                        The "label_names" key matches the pixel values with the label names. The "category" key indicates the labels belonging in each category. 
    indoor_categories (list): list containing the names for the indoor categories (here: 'thing_indoor' and 'stuff_indoor')
    
    
    Returns:
    float: Percentage of 'indoor' pixels in the image if the image contains any 'indoor' pixels.
    float: Percentage of 'indoor' pixels for images containing at least one 'indoor' pixel.
    None: If the image does not contain any 'indoor' pixels, or if it is empty.
   
    """
    unique_labels, counts = np.unique(annotation_img, return_counts=True)
    count_indoor_pixel = 0 #initialisation of the counts
    total_pixels = np.sum(counts)
    
    for label,count in zip(unique_labels, counts):
        for category in indoor_categories:
            #if label is from the indoor category
            if (label) in labels_json["categories"][category] :
                count_indoor_pixel += count
    
    if total_pixels > 0:
        percentage = count_indoor_pixel / total_pixels *100

        indoor_pixels_percentage_all_images = percentage
    else:
        indoor_pixels_percentage_all_images = None
        
    if count_indoor_pixel > 0:
        indoor_pixels_percentage_indoor_images = percentage
    else:
        indoor_pixels_percentage_indoor_images = None
    
    return indoor_pixels_percentage_all_images,indoor_pixels_percentage_indoor_images
    
def run_EDA(annotations,label_config,plot_directory):
    """  This function runs the EDA
    
    Parameters:
    annotations (str): The path leading to the annotations (ie the dataset).
    label_config (str): The path leading to the label config (json)
    plot_directory (str): The path where all the figures have to be saved in
    
    Returns:
    tuple: A tuple containing various analysis results (useful for testing the function)
    """
    #creating the folder for saving the plots
    os.makedirs(plot_directory, exist_ok=True)

    #opening the json file      
    try :
        with open(label_config, 'r') as f:
            labels = json.load(f)
            
            data_found = False


            #initialisation:
            pixel_min=float('inf')
            pixel_max=float('-inf')
            lst_image_sizes = []

            # dictionaries storing the count of pixels for each label in each category, and the count of images containing each label in each category.
            pixel_label_counts = {key: {} for key in labels["categories"].keys()}
            image_label_counts = {key: {} for key in labels["categories"].keys()}
            for cat, label_ids in labels["categories"].items():
                for label_id in label_ids:
                    label_name = labels["label_names"].get(str(label_id))
                    pixel_label_counts[cat][label_name] = 0
                    image_label_counts[cat][label_name] = 0                
                

            lst_unlabelled_percentage_pixels = []

            #inspired from https://stackoverflow.com/questions/12595051/check-if-string-matches-pattern
            indoor_categories = [category for category in labels["categories"].keys() if re.match(r'.*indoor*.',category) ]
            lst_indoor_percentage_all_images = []
            lst_indoor_percentage_indoor_images = []

            #going through each annotation image of the dataset
            for annotation_file in tqdm(os.listdir(annotations),desc='Exploratory Data Analysis...'):
                annotation_path = os.path.join(annotations, annotation_file)
                try :
                    annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
                    if annotation_img is None: 
                        raise FileNotFoundError() #image not found
                    
                    #process the images
                    lst_image_sizes.append(images_size_exploration(annotation_img))
                    pixel_label_counts,image_label_counts,pixel_min,pixel_max = pixel_label_exploration(annotation_img,labels,pixel_label_counts,image_label_counts,pixel_min,pixel_max)
                    lst_unlabelled_percentage_pixels.append(unlabelled_pixels_analysis(annotation_img))
                    indoor_pixel_percent_all_images, indoor_pixel_percent_indoor_images = calculate_indoor_pixel_percentage(annotation_img,labels,indoor_categories)
                    if indoor_pixel_percent_all_images is not None:
                        lst_indoor_percentage_all_images.append(indoor_pixel_percent_all_images)
                    if indoor_pixel_percent_indoor_images is not None:
                        lst_indoor_percentage_indoor_images.append(indoor_pixel_percent_indoor_images)              

                    data_found = True
                    
                except FileNotFoundError:
                    print(f"Image not found: {annotation_file}")

            if data_found:
                if pixel_min==float('inf'):
                    pixel_min = None
                if pixel_max==float('inf'):
                    pixel_max = None
                print(f"The minimum pixel value is {pixel_min}, and the maximum pixel value is {pixel_max} (excluding 255)")
                
                #plotting all the figures:
                plot_hexbin(lst_image_sizes,plot_directory) #hexbin to represent the distribution of image sizes

                #creation of a bar graph for each category
                for category in labels["categories"].keys():
                    pixel_counts = pixel_label_counts.get(category, {})
                    image_counts = image_label_counts.get(category, {})
                    #ordering the counts by value, from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
                    pixel_ordered_counts = dict(sorted(pixel_counts.items(), key=lambda item:item[1]))
                    image_ordered_counts = {label: image_counts[label] for label in pixel_ordered_counts.keys()}
                    plot_label_distribution(pixel_ordered_counts,image_ordered_counts, category, plot_directory)


                plot_boxplot(plot_directory,lst_unlabelled_percentage_pixels,"Percentage of unlabelled pixels across all images")

                plot_boxplot(plot_directory,lst_indoor_percentage_indoor_images,"Percentage of 'indoor' pixels",lst_indoor_percentage_all_images)

                return (lst_image_sizes, pixel_label_counts, image_label_counts, pixel_min, pixel_max, lst_unlabelled_percentage_pixels, lst_indoor_percentage_all_images, lst_indoor_percentage_indoor_images)

            else:
                print (f"No data found in the folder {annotations}.")

                
    except FileNotFoundError:
        print(f"Folder {annotations} not found.")
        labels = None
    #inspired from geeksforgeeks.org/json-parsing-errors-in-python/
    except json.JSONDecodeError:
        print(f"Error: The file {label_config} is not a valid JSON.")
        labels = None

if __name__ == '__main__':
    
    #getting the images and the json file containing the label config
    current_directory = os.getcwd()
    
    #annotations = os.path.join(current_directory,'data', 'filtered_data', 'filtered_annotations')
    annotations = os.path.join(current_directory,'data', 'raw_data', 'annotations')
    
    label_config = os.path.join(current_directory,'data','label_config.json')

    #creation of a folder to save the figures
    plot_directory = os.path.join(current_directory, 'data', 'plots','raw_data')
    
    _ = run_EDA(annotations,label_config,plot_directory)
   