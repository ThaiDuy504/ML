from skimage.io import imread
import matplotlib.pyplot as plt
import os

def display_sample_images(data_path, subset="training_set", category="cats", num_images=5):
    category_path = os.path.join(data_path, subset, category)
    images = os.listdir(category_path)[:num_images]
    for img_name in images:
        img_path = os.path.join(category_path, img_name)
        img = imread(img_path)
        plt.imshow(img)
        plt.title(f"{subset} - {category} - {img_name}")
        plt.axis("off")
        plt.show()

data_path = "data/raw/"
display_sample_images(data_path, subset="training_set", category="cats")
display_sample_images(data_path, subset="training_set", category="dogs")
