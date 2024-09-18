from src.data_pre_processing.data_cleaning import *

class TestPreProcess:

    def test_not_matching_images(self):
        """Testin when the image and semantic map do not have the same dimmensions.
        It should return 0 according to the documentation."""
        image = np.array([[0, 5], [10, 100]], dtype=np.uint8) 
        size = (2,2)
        semantic_map =  np.array([[0], [2]], dtype=np.uint8) 
        returned_images = preprocess(image,semantic_map, size)
        assert returned_images== None

    def _test_size(self):
        """Testing that the size of normalisation is respected"""
        image = np.array([[0, 5,0,0,0], [10, 100,0,0,0]], dtype=np.uint8) 
        size = (2,2)
        semantic_map =  np.array([[0, 1,0,0,0], [2, 3,0,0,0]], dtype=np.uint8) 
        h,w,_ = preprocess(image,semantic_map, size).shape
        expected_h = 2
        expected_w = 2
         
        assert h== expected_h
        assert w == expected_w