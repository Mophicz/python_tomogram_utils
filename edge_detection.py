from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the two TIFF images as NumPy arrays
image1 = np.array(Image.open('/Volumes/homes/zengin/Projects/Internship_Pos_lab/Segmentation_fiji/images10/images10_IM.tif'))
image2 = np.array(Image.open('/Volumes/homes/zengin/Projects/Internship_Pos_lab/Segmentation_fiji/images10/images10_OM.tif'))

# Combine the images by taking the higher pixel value
combined_image = np.maximum(image1, image2)

# Save the combined image
output_image = Image.fromarray(combined_image)

# Visualization
plt.figure()
plt.title("Combined Image")
plt.imshow(combined_image, cmap="gray")
plt.show()
