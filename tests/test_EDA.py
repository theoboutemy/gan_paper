import numpy as np
from src.data_pre_processing.EDA import *

class TestImagesSizeExploration:
    """Class testing the test_images_size_exploration function"""

    
    def test_regular_situation(self):
        """Regular situation: creation of a grayscale image, dimmension 100*200"""
        img = np.zeros((100,200),dtype = np.uint8)
        w , h = images_size_exploration(img)
        assert (w,h) == (200,100)

    def test_empty_image(self):
        """Testing with an empty image"""
        img = np.array([], dtype=np.uint8)
        w , h = images_size_exploration(img)
        assert (w,h) == (0,0)
    
class TestPixelLabelExploration:
    """Class testing the pixel_label_exploration function"""
    
    
    def setup_method(self):
        """Function allowing to setup the common variables before each test.
        Min and max are initialized.
        labels_json defines 3 labels "tv", "sky", "tree" and 2 categories: "indoor" and "outdoor"
        """
        self.start_min = float('inf')
        self.start_max = float('-inf')
        self.labels_json = {"label_names": {"0": "tv", "1": "sky", "2": "tree"},
                            "categories": {"indoor": [0], "outdoor": [1, 2]}}
        self.pixel_label_counts = {'indoor': {'tv': 0}, 'outdoor': {'sky': 0, 'tree': 0}}
        self.image_label_counts = {'indoor': {'tv': 0}, 'outdoor': {'sky': 0, 'tree': 0}}

    def assert_results(self, returned_count_pixel, returned_count_image, returned_min, returned_max,
                       expected_count_pixel, expected_count_image, expected_min, expected_max):
        """Function checking if a test is passed or not, by comparing the expected with the returned values.
        """
        assert expected_count_image == returned_count_image
        assert expected_count_pixel == returned_count_pixel
        assert expected_min == returned_min
        assert expected_max == returned_max

    def test_regular_situation(self):
        """
        Function testing a regular situation, where the image countains all the labels at least once.
        """
        img = np.array([[1, 2], [0, 1]], dtype=np.uint8)
        returned_count_pixel, returned_count_image, returned_min, returned_max = pixel_label_exploration(
            img, self.labels_json, self.image_label_counts, self.pixel_label_counts, self.start_min, self.start_max
        )

        expected_count_pixel = {"indoor": {"tv": 1}, "outdoor": {"tree": 1, "sky": 2}}
        expected_count_image = {"indoor": {"tv": 1}, "outdoor": {"tree": 1, "sky": 1}}
        expected_min, expected_max = 0, 2

        self.assert_results(returned_count_pixel, returned_count_image, returned_min, returned_max,
                            expected_count_pixel, expected_count_image, expected_min, expected_max)

    def test_label_not_represented(self):
        """
        In this test, the label "tv" is not represented in the image. It should still be counted as 0.
        """
        img = np.array([[1, 2], [1, 1]], dtype=np.uint8)
        returned_count_pixel, returned_count_image, returned_min, returned_max = pixel_label_exploration(
            img, self.labels_json, self.image_label_counts, self.pixel_label_counts, self.start_min, self.start_max
        )

        expected_count_pixel = {"indoor": {"tv": 0}, "outdoor": {"tree": 1, "sky": 3}}
        expected_count_image = {"indoor": {"tv": 0}, "outdoor": {"tree": 1, "sky": 1}}
        expected_min, expected_max = 1, 2

        self.assert_results(returned_count_pixel, returned_count_image, returned_min, returned_max,
                            expected_count_pixel, expected_count_image, expected_min, expected_max)

    def test_empty_image(self):
        """Testing with an empty image."""
        img = np.array([], dtype=np.uint8)
        returned_count_pixel, returned_count_image, returned_min, returned_max = pixel_label_exploration(
            img, self.labels_json, self.image_label_counts, self.pixel_label_counts, self.start_min, self.start_max
        )

        expected_count_pixel = {"indoor": {"tv": 0}, "outdoor": {"tree": 0, "sky": 0}}
        expected_count_image = {"indoor": {"tv": 0}, "outdoor": {"tree": 0, "sky": 0}}
        expected_min, expected_max = float('inf'), float('-inf')

        self.assert_results(returned_count_pixel, returned_count_image, returned_min, returned_max,
                            expected_count_pixel, expected_count_image, expected_min, expected_max)

class TestUnlabelledPixelAnalysis:
    """Class testing the pixel_label_exploration function"""
    
    def test_regular_situation(self):
        img = np.array([[1, 255], [0, 1]], dtype=np.uint8)
        expected_percentage = 1/4*100
        returned_percentage = unlabelled_pixels_analysis(img)
        assert expected_percentage == returned_percentage
        
    def test_no_unlabelled_pixel(self):
        """Here, the image does not contain any unlabelled pixel"""
        img = np.array([[1, 2], [0, 1]], dtype=np.uint8)
        expected_percentage = 0
        returned_percentage = unlabelled_pixels_analysis(img)
        assert expected_percentage == returned_percentage
    
    def test_all_unlabelled_pixel(self):
        """Here, all the pixels of the image are 'unlabelled' """
        img = np.array([[255, 255], [255, 255]], dtype=np.uint8)
        expected_percentage = 100
        returned_percentage = unlabelled_pixels_analysis(img)
        assert expected_percentage == returned_percentage
    
    def test_empty_image(self):
        """Testing with an empty image"""
        img = np.array([], dtype=np.uint8)
        expected_percentage = None
        returned_percentage = unlabelled_pixels_analysis(img)
        assert expected_percentage == returned_percentage


class TestCalculateIndoorPixel:
    """Class testing the calculate_indoor_pixel_percentage function"""
    def setup_method(self):
        """Function allowing to setup the common variables before each test.
        labels_json defines 3 labels "tv", "sky", "tree" and 2 categories: "indoor" and "outdoor"
        "indoor_categories" is a list containing the string "indoor"
        """
        self.labels_json = {"label_names": {"0": "tv", "1": "sky", "2": "tree"},
                            "categories": {"indoor": [0], "outdoor": [1, 2]}}
        self.indoor_categories= ["indoor"]                    
    
    def test_regular_situation(self):
        """Test: image containing two indoor category pixels"""
        img = np.array([[0, 1], [0, 2]], dtype=np.uint8)
        indoor_pixels_percentage_all, indoor_pixels_percentage_indoor  = calculate_indoor_pixel_percentage(img,self.labels_json,self.indoor_categories)
        expected_percentage_all = 2/4*100
        expected_percentage_indoor = 2/4*100
        assert expected_percentage_all==indoor_pixels_percentage_all
        assert expected_percentage_indoor==indoor_pixels_percentage_indoor

        
        
    def test_empty_image(self):
        """Test with an empty image"""
        img = np.array([], dtype=np.uint8)
        indoor_pixels_percentage_all, indoor_pixels_percentage_indoor  = calculate_indoor_pixel_percentage(img,self.labels_json,self.indoor_categories)
        expected_percentage_all = None
        expected_percentage_indoor = None
        assert expected_percentage_all==indoor_pixels_percentage_all
        assert expected_percentage_indoor==indoor_pixels_percentage_indoor
        
    def test_no_indoor_pixel(self):
        """Test with an image that does not contain any indoor pixel: it should return None as we are 
        interested in observing the probability that an image is mostly "indooor" if it contains at least one indoor pixel"""
        img = np.array([[1, 1], [2, 1]], dtype=np.uint8)
        indoor_pixels_percentage_all, indoor_pixels_percentage_indoor  = calculate_indoor_pixel_percentage(img,self.labels_json,self.indoor_categories)
        expected_percentage_all = 0
        expected_percentage_indoor = None
        assert expected_percentage_all==indoor_pixels_percentage_all
        assert expected_percentage_indoor==indoor_pixels_percentage_indoor
        
class TestEDA:
    """Class testing the EDA run function. Mainly testing what happens if a folder is empty or if the 
    json file is not valid"""
    def setup_method(self):
        """Function allowing to setup the common variables before each test.
        """
        self.current_directory = os.getcwd()
        self.annotations_dir = os.path.join(self.current_directory,"tests","test_files","annotations_folder_test")
        #this folder contains 3 images : [[3 ,3],[3, 3]] and [2, 1],[255, 2]] and [[0, 255,2],[1, 2,255]]
        os.makedirs(self.annotations_dir, exist_ok=True)
        #to modify the images of the test set :
        img1 = np.array([[3 ,3],[3, 3]], dtype=np.uint8)
        img2 = np.array([[2, 1],[255, 2]], dtype=np.uint8)
        img3 = np.array([[0, 255,2],[1, 2,255]], dtype=np.uint8)
        cv2.imwrite(os.path.join(self.annotations_dir,'img1.png'), img1)
        cv2.imwrite(os.path.join(self.annotations_dir,'img2.png'), img2)
        cv2.imwrite(os.path.join(self.annotations_dir,'img3.png'), img3)

        self.label_config_path = os.path.join(self. current_directory, 'tests','test_files', 'label_setup.json')
        self.plot_directory = os.path.join(self. current_directory,'tests','test_files', 'plots')
    
    #inspired from https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html
    def test_annotations_not_exist(self,capsys):
        annotations_dir = os.path.join(self. current_directory, 'nothing')
        _ =run_EDA(annotations_dir,self.label_config_path,self.plot_directory)
        captured = capsys.readouterr()
        #strip to remove the spaces
        outputs = captured.out.strip()
        assert outputs == f"Folder {annotations_dir} not found.".strip()
    
    def test_json_not_valid(self,capsys):
        label_config_path = os.path.join(self. current_directory,'tests','test_files', 'bad.json')
        _ = run_EDA(self.annotations_dir,label_config_path,self.plot_directory)
        captured = capsys.readouterr()
        #strip to remove the spaces
        outputs = captured.out.strip()
        assert outputs == f"Error: The file {label_config_path} is not a valid JSON.".strip()
    
    def test_regular_situation(self):
        """The testing folder  contains 3 images : [[3 ,3],[3, 3]] and [2, 1],[255, 2]] and [[0, 255,2],[1, 2,255]]"""
        returned_elements = run_EDA(self.annotations_dir,self.label_config_path,self.plot_directory)
        (lst_image_sizes, pixel_label_counts, image_label_counts, pixel_min, pixel_max, lst_unlabelled_percentage_pixels, lst_indoor_percentage_all_images, lst_indoor_percentage_indoor_images) = returned_elements
        #[[3 ,3],[3, 3]] and [2, 1],[255, 2]] and [[0, 255,2],[1, 2,255]]
        
        expected_lst_image_sizes = [(2, 2),(2, 2), (3, 2)]
        expected_pixel_label_counts = {'indoor': {'tv': 4}, 'outdoor': {'person': 1, 'sky': 2, 'tree': 4}}
        expected_image_label_counts = {'indoor': {'tv': 1}, 'outdoor': {'person': 1,'sky': 2, 'tree': 2}}
        expected_pixel_min = 0
        expected_pixel_max = 3
        expected_lst_unlabelled_percentage_pixels = [0,1/4*100,2/6*100]
        expected_lst_indoor_percentage_all_images = [100,0,0]
        expected_lst_indoor_percentage_indoor_images = [100]
        
 
        assert sorted(lst_image_sizes) == sorted(expected_lst_image_sizes)
        assert pixel_label_counts == expected_pixel_label_counts
        assert image_label_counts==expected_image_label_counts
        assert pixel_min==expected_pixel_min
        assert pixel_max ==expected_pixel_max
        assert sorted(lst_unlabelled_percentage_pixels) == sorted(expected_lst_unlabelled_percentage_pixels)
        assert sorted(lst_indoor_percentage_all_images) == sorted(expected_lst_indoor_percentage_all_images)
        assert sorted(lst_indoor_percentage_indoor_images)==sorted(expected_lst_indoor_percentage_indoor_images)
