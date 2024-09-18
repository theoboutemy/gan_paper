import os
from tqdm import tqdm
import shutil

def move_directories(source, destination, description):
    """Function moving the files of a source directory to a destination directory.
    
    Args:
    source (str): Path to source directory.
    destination (str): Path to destination directory.
    description (str): Description of the action performed.
    """
    for filename in tqdm(os.listdir(source), desc=description):
        shutil.move(os.path.join(source, filename), destination)


if __name__ == "__main__":
    
    try :
        current_directory = os.getcwd()

        #defining the directories
        annotations_val = os.path.join(current_directory,'data', 'dataset', 'annotations',"val2017")
        annotations_train = os.path.join(current_directory,'data', 'dataset', 'annotations',"train2017")
        images_val = os.path.join(current_directory,'data', 'dataset', 'images',"val2017")
        images_train = os.path.join(current_directory,'data', 'dataset', 'images',"train2017")
        

        merging_folder_images = os.path.join(current_directory,'data', 'raw_data', 'images')
        merging_folder_annotations = os.path.join(current_directory,'data', 'raw_data', 'annotations')
        os.makedirs(merging_folder_images, exist_ok=True)
        os.makedirs(merging_folder_annotations, exist_ok=True)

        #merging training and validation set
        move_directories(annotations_val, merging_folder_annotations, "Merging folders : 1/4")
        move_directories(annotations_train, merging_folder_annotations, "Merging folders : 2/4")
        move_directories(images_val, merging_folder_images, "Merging folders : 3/4")
        move_directories(images_train, merging_folder_images, "Merging folders : 4/4")

        #removing the initial directory as the images have been moved.
        shutil.rmtree(os.path.join(current_directory,'data', 'dataset'))
    
    except FileNotFoundError :
        print (f"The folders containing the images and semantic map do not exist. Download the data first. ")