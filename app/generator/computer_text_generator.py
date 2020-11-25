import random
from app.generator.utils import *
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
from app.utils import image_utils


# 旋转函数
def random_rotate(img):
    ''' ______________
        |  /        /|
        | /        / |
        |/________/__|
        旋转可能有两种情况，一种是矩形，一种是平行四边形，
        但是传入的points，就是4个顶点，
    '''

    w, h = img.size
    center = (w // 2, h // 2)

    degree = random.uniform(-30, 30)  # 随机旋转0-8度
    print("旋转度数:%f" % degree)
    return img.rotate(degree, center=center, expand=1)


# 因为英文、数字、符号等ascii可见字符宽度短，所以要计算一下他的实际宽度，便于做样本的时候的宽度过宽
def caculate_text_shape(text, font):
    # 获得文字的offset位置
    offsetx, offsety = font.getoffset(text)
    # 获得文件的大小,font.getsize计算的比较准
    width, height = font.getsize(text)

    width = width  # - offsetx
    height = height  # - offsety

    return width, height


def create_sentence_image(random_word, font, font_color):
    # 随机选取10个字符，是从info.txt那个词库里，随机挑的长度的句子
    width, height = caculate_text_shape(random_word, font)
    words_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(words_image)
    # 注意下，下标是从0,0开始的，是自己的坐标系
    draw.text((0, 0), random_word, fill=font_color, font=font)

    ############### PIPELINE ###########################
    words_image = random_rotate(words_image)
    return words_image


class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, image, text, font, text_color, font_size, orientation, space_width):
        """

        :param image:
        :param text:
        :param font:
        :param text_color:
        :param font_size:
        :param orientation:
        :param space_width:
        :return:
        """
        # TODO 水平和垂直的在同一张图片上生成
        if orientation == 0:
            return cls.__generate_horizontal_text(image, text, font, text_color, font_size, space_width)
        elif orientation == 1:
            return cls.__generate_vertical_text(image, text, font, text_color, font_size, space_width)
        else:
            raise ValueError("Unknown orientation " + str(orientation))

    @classmethod
    def __generate_horizontal_text(cls, bg_path, texts, font, text_color, font_size, space_scale):
        """
            生成水平文本
        :param bg_path: 背景图片地址
        :param texts:
        :param font:
        :param text_color:
        :param font_size:
        :param space_width:
        :return:
        """
        # 字体读取
        image_font = ImageFont.truetype(font=font, size=font_size)
        space_width = image_font.getsize(' ')[0] * space_scale

        # 背景图片
        txt_img = image_utils.read(bg_path)
        img_w, img_h = txt_img.size
        labelLists = []

        for text in texts:
            words = text.split(' ')
            # 计算每个字宽度
            words_width = [image_font.getsize(w)[0] for w in words]
            # 宽度+字间距，计算字符串总长度
            text_width = sum(words_width) + int(space_width) * (len(words) - 1)
            # 高度选所有文字中最高的一个
            text_height = max([image_font.getsize(w)[1] for w in words])
            # TODO 没有任何旋转倾斜仿射变换等，如果本身文本块大于背景也不能贴
            if text_width > img_w or text_height > img_h:
                print("文字太大：", text_width, text_height)
                continue
            can_add = False
            # 遍历5次找一个能贴图的地方
            for i in range(5):
                x0 = random.randint((text_width // 2) + 1, img_w - (text_width // 2) - 1)
                y0 = random.randint((text_height // 2) + 1, img_h - (text_height // 2) - 1)
                # 如果超出边缘了就不能再贴了
                if x0 + text_width > img_w or y0 + text_height > img_h:
                    continue

                box1label = [x0, y0, x0 + text_width, y0 + text_height, text]
                can_add = True  # 是否能粘贴
                if len(labelLists) > 0:
                    # 第一张小图随便贴,之后要判断是否重叠
                    # 看是否和之前框的重叠，如果重叠就不贴了。（TODO 可以多尝试几次，比如5次失败再放弃）
                    for box2label in labelLists:
                        if mat_inter(box1label, box2label):
                            can_add = False
                            break
                if can_add:
                    break
            if can_add:
                labelLists.append(box1label)
                colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
                c1, c2 = colors[0], colors[-1]
                # TODO 这里是一个参数
                fill = (
                    random.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
                    random.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
                    random.randint(min(c1[2], c2[2]), max(c1[2], c2[2]))
                )
                # fill = (0, 0, 0)
                # TODO 一个字一个字的贴，这特么有问题吧，为什么留那么大空隙
                # TODO 旋转之后再贴图？ 那底色怎么处理
                word_img = create_sentence_image(text, image_font, fill)
                txt_img.paste(word_img, (x0, y0), word_img)
            else:
                print("没有找到可以粘贴的框")
        return txt_img, labelLists

    @classmethod
    def __generate_vertical_text(cls, bg_path, texts, font, text_color, font_size, space_scale):
        image_font = ImageFont.truetype(font=font, size=font_size)
        txt_img = image_utils.read(bg_path)
        txt_draw = ImageDraw.Draw(txt_img)
        img_w, img_h = txt_img.size
        labelLists = []
        for text in texts:
            print("文字：", text)
            # 这里面就是用空格分开的
            space_height = int(image_font.getsize(' ')[1] * space_scale)

            char_heights = [image_font.getsize(c)[1] if c != ' ' else space_height for c in text]
            text_width = max([image_font.getsize(c)[0] for c in text])
            text_height = sum(char_heights)

            if text_width > img_w or text_height > img_h:
                continue

            can_add = False
            # 遍历5次找一个能贴图的地方
            for i in range(5):
                x0 = random.randint((text_width // 2) + 1, img_w - (text_width // 2) - 1)
                y0 = random.randint((text_height // 2) + 1, img_h - (text_height // 2) - 1)

                if text_width > img_w or text_height > img_h:
                    continue
                box1label = [x0, y0, x0 + text_width, y0 + text_height, text]
                can_add = True
                if len(labelLists) > 0:
                    for box2label in labelLists:
                        if mat_inter(box1label, box2label):
                            can_add = False
                            break
                if can_add:
                    break

            if can_add:
                # print("可以粘贴")
                labelLists.append(box1label)
                colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
                c1, c2 = colors[0], colors[-1]

                fill = (
                    random.randint(c1[0], c2[0]),
                    random.randint(c1[1], c2[1]),
                    random.randint(c1[2], c2[2])
                )
                for i, c in enumerate(text):
                    # 单字往下贴，我不要这种 也可以贴，贴在小图上，然后把小图贴到大图上。也可以。
                    txt_draw.text((x0, y0 + sum(char_heights[0:i])), c, fill=fill, font=image_font)

        return txt_img, labelLists
