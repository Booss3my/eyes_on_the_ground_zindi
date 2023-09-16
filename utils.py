
import matplotlib.pyplot as plt
def display_image_tensor(img_tensor):
    img_array = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_array)
    plt.axis('off') 
    plt.show()