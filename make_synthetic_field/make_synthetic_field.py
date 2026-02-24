import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def convert_image_to_npy(ll,hl, path):
    
    # Load image
    image = Image.open(path).convert('L')  # Convert to grayscale
    
    # Apply binary threshold
    bw = image.point(lambda x: 255 if x > 80 else 0, mode='1')
    
    
    im1= ll*np.ones_like(image)
    im1[bw] = hl


    np.save("synthetic_field",im1)
    

    plt.imshow(im1, cmap='bwr_r', extent= [617.33, 620.64, 6968.75, 6972.98])
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    
    plt.colorbar(label='Phase Velocity')
    plt.savefig('True_synthetic_field.png')
    plt.show()


if __name__ == '__main__':
    
    ll = 2    # low velocity level
    hl = 3    # high velocity level
    image_path = Path(__file__).resolve().parent / "syn_field.png"
    convert_image_to_npy(ll, hl, image_path)
