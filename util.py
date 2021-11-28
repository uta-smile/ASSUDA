from __future__ import print_function
import numpy as np
from PIL import Image
import numpy as np


def save_image(img, path, name, iteration, is_adv=False):
	img = tensor2im(img)
	img = Image.fromarray(img)
	if is_adv:
		img.save('%s/%s_%s_color_adv.png' % (path, name, str(iteration)))
	else:
		img.save('%s/%s_%s_color.png' % (path, name, str(iteration)))
