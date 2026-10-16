from PIL import Image
import math


def padded_bounds(bbox, image_size, padding):
    """Expand by a fraction of width/height per side, then clip to the page."""
    width, height = image_size
    x1, y1, x2, y2 = map(float, bbox)
    assert all(math.isfinite(v) for v in [x1, y1, x2, y2])
    assert width > 0 and height > 0 and x2 > x1 and y2 > y1 and padding >= 0
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    bounds = (max(0, min(width, math.floor(x1-dx))), max(0, min(height, math.floor(y1-dy))),
              max(0, min(width, math.ceil(x2+dx))), max(0, min(height, math.ceil(y2+dy))))
    assert bounds[2] > bounds[0] and bounds[3] > bounds[1], 'Empty padded crop after clipping.'
    return bounds


def prepare_crop(image, bbox, padding, size):
    """Fit the entire padded RGB crop onto a white square; never center-crop away its borders."""
    bounds = padded_bounds(bbox, image.size, padding)
    crop = image.convert('RGB').crop(bounds)
    scale = min(size/crop.width, size/crop.height)
    resized = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))),
                          Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(resized, ((size-resized.width)//2, (size-resized.height)//2))
    return canvas, bounds
