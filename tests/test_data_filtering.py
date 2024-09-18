from src.data_pre_processing.data_filtering import *

class TestImageToKeep:
    def setup_method(self):
        self.current_directory = os.getcwd()
        self.label_config = os.path.join(self.current_directory, 'tests','test_files', 'label_setup.json')
        #defines  defines 4 labels "tv:0", "sky:1", "tree:2", "person:3" and 2 categories: "indoor" and "outdoor"

    
    def test_true(self):
        """Testing with an image to keep"""
        img = np.random.choice([1, 2], size=(256, 256))
        #no human, no indoor pixel
        returned_bool = image_to_keep(img,self.label_config)
        assert returned_bool == True
        
    def test_false_human(self):
        """Testing with an image containing a human"""
        img = np.random.choice([1, 2], size=(256, 256), p=[0.5, 0.5])

        # Choisir une position aléatoire pour l'élément égal à 3
        x, y = np.random.randint(0, 256, size=2)
        img[x, y] = 0
         #with human
        returned_bool = image_to_keep(img,self.label_config)
        assert returned_bool == False
    
    def test_false_size(self):
        """Testing with an image with dimensions < 256*256"""
        img = np.array([[3, 3], [1, 1]], dtype=np.uint8) 
        returned_bool = image_to_keep(img,self.label_config)
        assert returned_bool == False
        
    def test_false_indoor_pixel(self):
        """Testing with an image containing more than 5 % indoor pixels"""
        img = np.full((256, 256), 3, dtype=np.uint8)


        returned_bool = image_to_keep(img,self.label_config)
        assert returned_bool == False       
