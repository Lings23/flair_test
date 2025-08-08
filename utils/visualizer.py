import shap
import numpy as np
import copy 
import os
class Visualizer():
    
    def __init__(self, shap_util):
        self.shap_util = shap_util
        
    def plot_shap_images(shap_indices, shap_images):
        """
        Plot sample images and their target labels
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i, idx in enumerate(shap_indices):
            plt.subplot(3,4,i+1)
            plt.tight_layout()
            plt.imshow(shap_images[idx][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(self.targets[idx]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
        
    def plot_shap_values(self, shap_v, file=None, indices=None):
        """
        Plot SHAP values and image
        :param shap_values: name of file
        :type shap_values: Tensor
        :param file: name of file
        :type file: os.path
        """
        import matplotlib.pyplot as plt
        images = self.shap_util.shap_images[indices] if indices else self.shap_util.shap_images
            
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_v]
        test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
        if file:
            print("make picture")
            shap.image_plot(shap_numpy, -test_numpy, show=False)
            if not os.path.exists(os.path.dirname(file)):
                print("make dir")
                os.makedirs(os.path.dirname(file))
            print("Make picture")
            plt.savefig(file)
        else: 
            shap.image_plot(shap_numpy, -test_numpy)
            
    def compare_shap_values(self, s_client, s_server, file=None, indices=None):
        """
        Plot SHAP values and image
        :param shap_values: name of file
        :type shap_values: Tensor
        :param file: name of file
        :type file: os.path
        """
        import matplotlib.pyplot as plt
        images = self.shap_util.shap_images[indices] if indices else self.shap_util.shap_images
        shap_subtract = np.subtract(s_client, s_server)
        compare_shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_subtract]
        compare_test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
        if file:
            shap.image_plot(compare_shap_numpy, -compare_test_numpy, show=False)
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            plt.savefig(file)
        else: 
            shap.image_plot(compare_shap_numpy, -compare_test_numpy)
            
    def compare_normed_shap_values(self, s_client, s_server, file=None, indices=None):
        """
        Plot SHAP values and image
        :param shap_values: name of file
        :type shap_values: Tensor
        :param file: name of file
        :type file: os.path
        """
        import matplotlib.pyplot as plt
        images = self.shap_util.shap_images[indices] if indices else self.shap_util.shap_images
        shap_subtract = np.subtract(s_client, s_server)
        norms = np.linalg.norm(shap_subtract)
        compare_shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_subtract/norms]
        compare_test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
        if file:
            shap.image_plot(compare_shap_numpy, -compare_test_numpy, show=False)
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            plt.savefig(file)
        else: 
            shap.image_plot(compare_shap_numpy, -compare_test_numpy)