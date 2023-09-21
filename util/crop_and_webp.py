import os
from PIL import Image

# Crop images to resolution 128x128 and
# export both JPEG and WebP formats

input_dir = ''
output_dir = '../data/CelebA_jpg/image'

def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name_list = sorted(os.listdir(input_dir))
    for name in name_list:
        prefix = name.split('.')[0]
        input_path = os.path.join(input_dir, name)
        output_jpeg_path = os.path.join(output_dir, (prefix + '.jpg'))
        output_webp_path = os.path.join(output_dir, (prefix + '.webp'))
        
        img = Image.open(input_path)
        out_img = center_crop(img, 128, 128)
        out_img.save(output_jpeg_path)
        out_img.save(output_webp_path)
