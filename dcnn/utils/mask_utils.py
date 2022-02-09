import torch
import numpy as np
import cv2


def compute_closest_point(mask, the_point):

    the_x = the_point[0]
    the_y = the_point[1]
    mask_size = mask.size()
    x_coords = torch.tensor(list(range(mask_size[1])), device=mask.device) + torch.ones(mask_size[1], device=mask.device)
    y_coords = torch.tensor(list(range(mask_size[0])), device=mask.device) + torch.ones(mask_size[0], device=mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
    masked_x = torch.masked_select(x_grid, mask)
    masked_y = torch.masked_select(y_grid, mask)
    xdiffs = torch.sub(masked_x, the_x)
    ydiffs = torch.sub(masked_y, the_y)
    dists = torch.square(xdiffs) + torch.square(ydiffs)
    closest_idx = torch.argmin(dists).item()
    # print( (dists == torch.min(mask_dists)).nonzero() )

    # print(y_grid.shape)
    return (masked_x[closest_idx].item(), masked_y[closest_idx].item())


# mask is torch.Tensor type
def get_mask_centroid(mask):
    
    mask_size = mask.size()
    x_coords = torch.tensor(list(range(mask_size[1])), device=mask.device) + torch.ones(mask_size[1], device=mask.device)
    y_coords = torch.tensor(list(range(mask_size[0])), device=mask.device) + torch.ones(mask_size[0], device=mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords)

    mass = mask.sum()
    x = torch.floor_divide( (x_grid * mask).sum(), mass )
    y = torch.floor_divide( (y_grid * mask).sum(), mass )

    return (x.item(), y.item())


def compute_masks_iou(detection_mask, object_mask, detection_centroid=None):
    
    object_centroid = self.get_mask_centroid(object_mask)
    if detection_centroid is None:
        detection_centroid = self.get_mask_centroid(detection_mask)
    translation = (object_centroid[0] - detection_centroid[0], object_centroid[1] - detection_centroid[1])
    translated_detection = self.translate_and_crop_mask(detection_mask, translation)
    # show_mask(detection_mask, 'detection')
    # show_mask(object_mask, 'object')
    # show_mask(translated_detection, 'translated detection')
    intersection = translated_detection * object_mask
    union = translated_detection + object_mask
    union = union > 0
    return torch.true_divide(intersection.sum(), union.sum()).item()


def translate_and_crop_mask(mask, translation_vector):

    [height, width] = mask.size()
    (dx, dy) = translation_vector
    dx = int(dx)
    dy = int(dy)
    if dx >= 0:
        leftpad = dx
        rightpad = 0
    else:
        leftpad = 0
        rightpad = -dx
    if dy >= 0:
        toppad = dy
        bottompad = 0
    else:
        toppad = 0
        bottompad = -dy
    pad = torch.nn.ConstantPad2d((leftpad, rightpad, toppad, bottompad), 0)
    result = pad(mask)[bottompad : height+bottompad, rightpad : width+rightpad]
    return result


def show_mask(mask, text='mask'):

    mask = mask.clone().cpu().detach().numpy()
    mask = np.uint8(mask)*255
    cv2.imshow(text, mask)