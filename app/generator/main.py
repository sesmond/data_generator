#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   : 生成样本
@File    :   main.py    
@Author  : vincent
@Time    : 2020/11/18 3:05 下午
@Version : 1.0 
'''
import random

from app.generator.data_generator import generate
from app.utils import file_utils
from multiprocessing import Process, Pool
import logging

logger = logging.getLogger(__name__)


def init_logger():
    logger.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logger.INFO,
        handlers=[logger.StreamHandler()])


# 生成一张背景图，大小随机
def create_backgroud_image(bground_list):
    # 从背景目录中随机选一张"纸"的背景
    bground_choice = random.choice(bground_list)

    # return image, width, height
    return random_image_size(bground_choice)


# 随机裁剪图片的各个部分
def random_image_size(image):
    while True:
        # 产生随机的大小
        height = random.randint(MIN_BACKGROUND_HEIGHT, MAX_BACKGROUND_HEIGHT)
        width = random.randint(MIN_BACKGROUND_WIDTH, MAX_BACKGROUND_WIDTH)

        # 高度和宽度随机后，还要随机产生起始点x,y，但是要考虑切出来不能超过之前纸张的大小，所以做以下处理：
        size = image.size
        x_scope = size[0] - width
        y_scope = size[1] - height
        if x_scope < 0: continue
        if y_scope < 0: continue
        x = random.randint(0, x_scope)
        y = random.randint(0, y_scope)

        # logger.debug("产生随机的图片宽[%d]高[%d]",width,height)
        image = image.crop((x, y, x + width, y + height))
        # logger.debug("剪裁图像:x=%d,y=%d,w=%d,h=%d",x,y,width,height)
        return image, width, height


def generate_one(img_name):
    # TODO
    # 1. 字体 一张图片里要有几种字体，不要太多。而且要有个比例。
    # 2. 文字密度，有比较近也要有比较远的，要比较符合正常的
    # 3. 文字方向，要有横向的，也要有竖向的，并且做一下倾斜。还有其他变形什么的
    # 4. 文字密度为多少时坐标为一个框，密度为多少时给分开。
    # 5. 生成负样本，一些连在一起的虚线/直线不作为检测对象
    # 6. 英文，数字，日期，空格等。 其实最好的还是
    # 字体 1-3种，以其中一种为主，字形字号。
    # 字号也是 做成不一样大的吧。
    

    pass


def generate_batch(p_no, output_path, count=1000):
    # 1. 先读取所有背景图片
    bg_imgs = file_utils.get_files("config/background")
    # 2. 加载所有字体
    fonts = file_utils.get_files("config/fonts/cn", ["ttf", "ttc", "TTF", "TTC"])
    # 3.
    # 2.
    for i in range(0, count):
        logger.info("进程：%r,总%r第%r,开始处理", p_no, count, i)
        #TODO
        generate(i, img,
                 random.sample(strings, random.randint(1, 40)),
                 fonts[random.randrange(0, len(fonts))],
                 args.output_dir,
                 args.extension,
                 args.width,
                 args.text_color,
                 args.orientation,
                 args.space_width,
                 args.font_size)
    logger.info("进程：%r,处理结束。", p_no)
    return


def generate_process(output_path, worker):
    pool = Pool(processes=worker)
    for i in range(0, worker):
        pool.apply_async(generate_batch, args=(i, output_path))
    pool.close()
    pool.join()


if __name__ == '__main__':
    init_logger()
    generate_process()
