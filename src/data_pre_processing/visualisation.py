import matplotlib.pyplot as plt
import os
import numpy as np
import json,sys
import matplotlib.patches as mpatches



def plot_hexbin(image_sizes, plot_directory):
    """
    This function creates a figure (hexbin) to visualize the distribution of image sizes.
    
    Parameters:
    image_sizes (list): The list containing the sizes (width, height) to plot
    plot_directory (path): The path for saving the figure
    """
    plt.figure(figsize=(10, 8))
    widths, heights = zip(*image_sizes)
    plt.hexbin(widths, heights, gridsize=30, cmap='Accent', mincnt=1)
    plt.colorbar(label='Number of images')
    
    # Increase the font size of the title, labels, and ticks
    plt.title('Hexbin plot: images size', fontsize=20)
    plt.xlabel('Width', fontsize=20)
    plt.ylabel('Height', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.savefig(os.path.join(plot_directory, 'hexbin_image_size.png'))
    plt.close()


def plot_label_distribution(pixel_label_counts,image_label_counts, category_name, plot_directory):
    """  This function creates a figure (bar graph) to visualise the distribution of the labels.
    The bar graph displays the distribution of labels in terms of 
    the number of pixels and the number of images containing each label. 
    Warning: A different scale is applied for both

 
    Parameters:
    pixel_label_counts (dict): Dictionary to update. It stores the count of pixels for each label in each category.
    image_label_counts (dict): Dictionary to update. It stores the count of images containing each label in each category.
    category_name (str): name of the category represented in the data (eg: stuff_indoor)
    plot_directory (path): The path for saving the figure
    """
    #inspired from https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart
    position_x = np.arange(len(pixel_label_counts))

    bar_width = 0.4  # Width of the bars

    _, ax1 = plt.subplots(figsize=(25, 8))
    ax2 = ax1.twinx() #ax2 shares the same x axis than ax1

    bars1 = ax1.bar(position_x, pixel_label_counts.values(), color='orange', width=bar_width, edgecolor='black', label='Pixel Count')
    bars2=ax2.bar(position_x + bar_width, image_label_counts.values(), color='blue', width=bar_width, edgecolor='green', label='Image Count')

    ax1.legend([bars1], ['Pixel Count'], loc='upper left',fontsize=20)
    ax1.set_yscale('log') #log scale for the pixel count axis
    ax1.set_xticks(position_x + bar_width / 2) #label name position: in between the pixel and the image bars
    ax1.set_xticklabels(pixel_label_counts.keys(), rotation=90,weight='bold',fontsize=20)
    ax1.tick_params(axis='y', labelsize=18)
    
    ax2.tick_params(axis='y', labelsize=18)
    ax2.legend([bars2], ['Image Count'], loc='upper right',fontsize=18)
    
    plt.title(f'Labels distribution - {category_name}',weight='bold',fontsize=25)
    plt.xlabel('Label ID',weight='bold',fontsize=20)
    plt.ylabel('Number of occurrences',weight='bold',fontsize=20)
    plt.subplots_adjust(bottom=0.3, top=0.95)

    plt.savefig(os.path.join(plot_directory, f'labels_{category_name}.png'))
    plt.close()


def plot_boxplot(plot_directory,unlabelled_percentage_indoor,title,unlabelled_percentage_all = None):
    """  This function creates a box plot.
 
    Parameters:
    title (str): The title of the graph
    plot_directory (path): The path for saving the figure
    unlabelled_percentage_indoor (list): The list containing data to plot 
    unlabelled_percentage_all (list): optional. Plotting 2 box plots in this case

    """

    if unlabelled_percentage_all is not None:
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Box plot for indoor images
        axes[0].boxplot(unlabelled_percentage_indoor, sym='')
        axes[0].set_title('Indoor Images')
        axes[0].set_xticks([1])
        axes[0].set_xticklabels([''])
        axes[0].set_ylabel('Percentage')

        # Box plot for all images
        axes[1].boxplot(unlabelled_percentage_all, sym='')
        axes[1].set_title('All Images')
        axes[1].set_xticks([1])
        axes[1].set_xticklabels([''])
        axes[1].set_ylabel('Percentage')

        plt.suptitle(title)
        plt.savefig(os.path.join(plot_directory, f'boxplot_{title}.png'))
    else:
        plt.boxplot(unlabelled_percentage_indoor, sym='')
        plt.xticks([1], [''])  # So the graph is zoomed on the box plot, not on the outliers.
        plt.title(title)
        plt.ylabel('Percentage')
        plt.savefig(os.path.join(plot_directory, f'boxplot_{title}.png'))
    
    plt.close()
    
def plot_overlayed_image(image_name,combined_image,overlayed_annotations_directory,label_config,annotation_img,colours_labels):
    """This function plots the overlayed image by superimposing the semantic map onto the original image.

    Args:
        image_name (str): Name of the image.
        combined_image (np.ndarray): The overlayed image.
        overlayed_annotations_directory (str): The path to save the image to.
        label_config (str): The path to the label json file.
        annotation_img (np.ndarray): the image containing the annotations.
        colours_labels (dict): Dictionary associating each unique label of the image with a unique colour.
    """
    try: 
        color_patch = []
        _, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_image)
        ax.axis('off')
        with open(label_config, 'r') as f:
            labels = json.load(f)
            #inspired from https://www.tutorialspoint.com/how-to-manually-add-a-legend-with-a-color-box-on-a-matplotlib-figure#:~:text=We%20do%20this%20by%20creating,label%20of%20%22Example%20Legend%22.
            for label_id in (np.unique(annotation_img)):
                color_patch.append(mpatches.Patch(color=colours_labels[str(label_id)][:3]/255, label=labels["label_names"][str(label_id)]))
        plt.legend(handles = color_patch,loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(os.path.join(overlayed_annotations_directory,image_name))
        plt.close()

    #inspired from geeksforgeeks.org/json-parsing-errors-in-python/
    except json.JSONDecodeError:
        print(f"Error: The file {label_config} is not a valid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)