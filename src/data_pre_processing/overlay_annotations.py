import os
from tqdm import tqdm
import cv2
import json
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from src.data_pre_processing.visualisation import plot_overlayed_image
from random import sample 





def colours_label_dictionary( unique_labels_in_image):
    """This function creates a ditionary and associates a unique colour for each label given in argument.

    Args:
        unique_labels_in_image (lst) : List containing the unique label numbers of the image.

    Returns:
        dict : The dictionary associating a colour to a label number
    """
         
    #inspired from https://stackoverflow.com/questions/52196292/unable-to-found-rainbow-method-in-matplotlib
    color_indices = np.linspace(0, 1, len(unique_labels_in_image))
    colours = cm.rainbow(color_indices) * 255 #so the RGB colours are in the range (0,255)
    colours_label_dict = {str(label_nb) : colour for (label_nb,colour) in zip(unique_labels_in_image, colours)}
    return colours_label_dict

   
        
def overlay_annotations(annotations_directory,images_directory,label_config,overlayed_annotations_directory):
    """This function creates an overlayed image by superimposing the semantic map onto the original image.
    Each annotation image is associated with a color map, depending on the number of label it contains. 
    The original image is plotted in grayscale to prevent the colors from blending with the annotation colors,
    ensuring that the annotations are clearly visible without any color bias from the original image.


    Args:
        annotations_directory (str): The path to the semantic maps.
        images_directory (str): THe path to the images.
        label_config (str): The path to the label configuration file.
        overlayed_annotations_directory (str): The path to save the resulting overlayed images.

    Returns:
        None: The overlayed image is saved in the specified folder.
    """
    #creating the folder
    os.makedirs(overlayed_annotations_directory, exist_ok=True)
    
    
    #getting a unique colour 
    try: 
        images_list = os.listdir(images_directory) #all the images in the filtered_data folder
        #Random sample of 351 images from the dataset
        sample_size = 351
        sample_images = sample(images_list,sample_size)
        
        
        
        
        #for each image og the dataset
        for image_name in tqdm(sample_images, desc="Overlaying images and annotations"):
            image_path = os.path.join(images_directory,image_name)
            #warning : in the dataset, the images are in a .jpeg format, but the annotations are in a .png format
            image_no_extension, _ = os.path.splitext(image_name)
            annotations_path = os.path.join(annotations_directory, image_no_extension+".png")
        
            try:
                annotation_img = cv2.imread(annotations_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(image_path,cv2.IMREAD_COLOR)
                #the original image has to be turned in a grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                
                if annotation_img is None: 
                        raise FileNotFoundError("Annotations")
                if image is None: 
                        raise FileNotFoundError("Image")
                mask_image = np.zeros_like(image)

                colours_labels = colours_label_dictionary( np.unique(annotation_img))

                for label_id in (np.unique(annotation_img)):
                    label_colour = colours_labels[str(label_id)][:3]
                    mask = (annotation_img == label_id)
    
                    mask_image[mask] = label_colour
                    
                #combined_images = np.concatenate((image, mask_image), axis=1)
                combined_images = cv2.addWeighted(gray_image_bgr, 1, mask_image, 0.4, 0)

                plot_overlayed_image(image_name,combined_images,overlayed_annotations_directory,label_config,annotation_img,colours_labels)

                #cv2.imwrite(os.path.join(overlayed_annotations_directory,image_name), combined_images)

            except FileNotFoundError as not_found:
                print(f"File not found:{image_name} _ {not_found}")
            except Exception as e:
                print(f"An error occurred while processing {image_name}: {e}")
                
            
    except FileNotFoundError:
        print(f"Folder {annotations_directory} not found.")
        return False
    except Exception as e:
        print(f"An error occurred while processing the annotations directory: {e}")


if __name__ == '__main__':
    
    current_directory = os.getcwd()
    
    #defining the folders to the dataset.
    annotations = os.path.join(current_directory,'data', 'filtered_data', 'filtered_annotations')
    images = os.path.join(current_directory,'data', 'filtered_data', 'filtered_images')
    label_config = os.path.join(current_directory,'data','label_config.json')

    #defining the folder to save the overlayed images.
    overlayed_annotations_directory = os.path.join(current_directory, 'data', 'plots','overlayed_images')
    
    overlay_annotations(annotations,images,label_config,overlayed_annotations_directory)