from PIL import Image
import numpy as np
import torch

def downscale_image(image, target_size=(84, 84)):
    image = Image.fromarray(image)
    image = image.convert('L')

    image = image.resize(target_size, Image.BILINEAR)

    return np.array(image)

def update_frame_stack(stack, new_frame):
    new_stack = stack.clone()
    # Shift frames in the stack one position to the right
    new_stack[1:,:,:] = stack[:-1,:,:]

    # Insert the new frame at the beginning of the stack
    new_stack[0, :, :] = new_frame

    return new_stack