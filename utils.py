from PIL import Image
import numpy as np
import torch

def downscale_image(image, target_size=(84, 84)):
    image = Image.fromarray(image)
    image = image.convert('L')

    image = image.resize(target_size, Image.BILINEAR)

    return np.array(image)


def update_frame_stack(stack, new_frame):
    # Assuming new_frame is already on the correct device
    new_stack = torch.empty_like(stack)  # Create an empty tensor with the same shape and device as stack
    new_stack[0, :, :] = new_frame       # Insert the new frame at the beginning of the stack
    new_stack[1:, :, :] = stack[:-1, :, :]  # Shift the rest of the frames

    return new_stack
