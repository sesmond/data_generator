import os
import random

from PIL import Image, ImageFilter

from app.generator.computer_text_generator import ComputerTextGenerator
from app.utils import image_utils


def generate(index, image, text: list, font, out_dir, extension, width, text_color, orientation, space_width, size):
    """

    :param index: 索引
    :param image: 背景图片地址
    :param text: 要粘贴的文本（多条list）
    :param font: 字体路径
    :param out_dir: 输出路径
    :param extension: 生成图片后缀名
    :param width:
    :param text_color: 文本颜色
    :param orientation: 应该是朝向 0: Horizontal, 1: Vertical
    :param space_width: 字间距，多少空格（TODO）
    :param size:字体大小
    :return:
    """
    ##########################
    # Create picture of text #
    ##########################
    # TODO

    image, labelLists = ComputerTextGenerator.generate(image, text, font, text_color, size, orientation, space_width)

    #####################################
    # Generate name for resulting image #
    #####################################
    print("result labelLists:", len(labelLists), labelLists)
    if len(labelLists) != 0:
        image_name = '{}_{}.{}'.format(str(orientation), str(index), extension)
        # Save the image
        image.convert('RGB').save(os.path.join(out_dir, image_name))

        txt_name = '{}_{}.{}'.format(str(orientation), str(index), 'txt')
        f = open(os.path.join(out_dir, txt_name), 'w')

        for label in labelLists:
            [x0, y0, x1, y1, label] = label

            strline = str(int(x0)) + ',' + str(int(y0)) + ',' + str(int(x1)) + ',' + \
                      str(int(y0)) + ',' + str(int(x1)) + ',' + str(int(y1)) + ',' + str(int(x0)) + \
                      ',' + str(int(y1)) + ',' + label + '\n'

            f.write(strline)

        f.close()
