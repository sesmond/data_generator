import random
import re
import string
import requests

from bs4 import BeautifulSoup
import numpy as np

"""
生成样本字符串
"""

POSSIBILITY_PURE_NUM = 0.1  # 需要产生的纯数字 -->0.2
POSSIBILITY_PURE_ENG = 0.1  # 需要产生的英语 -->0.1
POSSIBILITY_DATE = 0.1  # 需要产生的纯日期 -->0.1
POSSIBILITY_SINGLE = 0.01  # 单字的比例
POSSIBILITY_SPECIAL = 0.2  # 特殊字符
POSSIBILITY_CONFUSION = 0  # 易混字的比列

MAX_LENGTH = 20  # 可能的最大长度（字符数）
MIN_LENGTH = 1  # 可能的最小长度（字符数）

MAX_SPECIAL_NUM = 5  # 特殊字符的个数
MAX_BLANK_NUM = 5  # 字之间随机的空格数量

MAX_GENERATE_NUM = 1000000000

POSSIBILITY_BLANK = 0.8  # 有空格的概率

# 易混字集合，用于增加这部分样本的比例
confusion_set = '的一是在不了和这为上个我以要他时来们生到作地于出就分对成会可主同工也能下过子说产种面而方后多学法得经十三之进着等部度家力里如水化\
高自二理起物现实加都两体当点从业本去把性好应它还因由其些然前外天政四那社义事平形相全表间样与各重新线内正心反你看原又么利比或但气向道命此变条只没结解问\
意建无系军很情者最立想已并提直题党程展五果料象员革位入常文次品式活设及管特件长求老头基资边流路级少图山统接知较将组见她手角根论运农指几九区强放决西被干\
做必战先回则任取据处队南给色光门即治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收改清己美再采转更风切打白教速花带场例真具万每目至\
达走积示议声报斗完类八离华确才科张马节米空况今集温传土许步群广石需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技半办\
青省列习响约般史感劳便团往酸历克何除消构府太准精值率族维划选标存候毛亲快效斯院查江眼王按格养易置派层片始却专状育厂京适属圆包火住调满县局照参红细引听该铁严'


def create_string(corpus):
    # 根据文章生成
    if corpus:
        random_word, _ = _get_random_text_from_corpus(corpus)
    else:
        # 根据字符集生成
        random_word, _ = _get_random_text(charset)
    # 首位不要空格，影响检测效果 TODO 中间也不要太多空格，有空格的一般要检测成两块
    random_word = random_word.strip()
    print(random_word)
    return random_word


# 加载字符集，charset.txt，最后一个是空格
# 为了兼容charset.txt和charset6k.txt，增加鲁棒性，改一下
# 先读入内存，除去
def _get_charset(charset_file):
    charset = open(charset_file, 'r', encoding='utf-8').readlines()
    charset = [ch.strip("\n") for ch in charset]
    charset = "".join(charset)
    charset = list(charset)
    if charset[-1] != " ":
        charset.append(" ")

    return charset


charset = _get_charset("config/dicts/cn.txt")


# 随机接受概率
def _random_accept(accept_possibility):
    return np.random.choice([True, False], p=[accept_possibility, 1 - accept_possibility])


def create_strings_from_dict(length, allow_variable, count, lang_dict):
    """
            Create all strings by picking X random word in the dictionnary
    :param length:
    :param allow_variable:
    :param count:
    :param lang_dict:
    :return:
    """

    dict_len = len(lang_dict)
    strings = []
    for _ in range(1, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            current_string += lang_dict[random.randrange(dict_len)][:-1]
            # TODO 加个空格有什么用
            # 这里加个空格，后面这个空格不会写进去，而是作为一个间隔符做字间距用的
            current_string += ' '
        strings.append(current_string[:-1])
    return strings


# 专门用来产生数字，可能有负数，两边随机加上空格
def _generate_num():
    num = random.uniform(-MAX_GENERATE_NUM, MAX_GENERATE_NUM)
    need_format = random.choice([True, False])
    if (need_format):
        return "{:,}".format(num)
    return str(num)


# 产生英文：50%是纯英文，20%是中英，20%是数字英文，10%是中、英、数字
def _generator_english(charactors):
    alphabeta = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW"
    num = "0123456789"
    s = ""
    length = random.randint(MIN_LENGTH, MAX_LENGTH)

    # if POSSIBILITY_PURE_ENG
    # E:english N:num C:Chinese
    options = ["E", "EN", "EC", "ENC"]
    opt = np.random.choice(options, p=[0.5, 0.2, 0.2, 0.1])

    def sample(charset, length):
        s = ""
        for i in range(length):
            j = random.randint(0, len(charset) - 1)
            s += charset[j]
        return s

    english = sample(alphabeta, length)
    if opt == "E":
        return english

    snum = sample(num, length)
    chinese = sample(charactors, length)

    if opt == "EN":
        all = list(english + snum)
        np.random.shuffle(all)
        return "".join(all[:length])

    if opt == "EC":
        all = list(english + chinese)
        np.random.shuffle(all)
        return "".join(all[:length])

    if opt == "ENC":
        all = list(english + snum + chinese)
        np.random.shuffle(all)
        return "".join(all[:length])

    raise ValueError("无法识别的Option：%s", opt)


# 专门用来产生日期，各种格式的
def _generate_date():
    import time
    now = time.time()

    date_formatter = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y年%m月%d日",
        "%Y%m%d ",
        "%y-%m-%d ",
        "%y/%m/%d ",
        "%y年%m月%d日",
        "%Y%m%d "
    ]

    _format = random.choice(date_formatter)

    _timestamp = random.uniform(0, now)

    time_local = time.localtime(_timestamp)

    return time.strftime(_format, time_local)


# 随机生成文字，长度是10-30个之间，512像素
def _generate_words(charset):
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    s = ""
    for i in range(length):
        j = random.randint(0, len(charset) - 1)
        s += charset[j]
    # if DEBUG: print("随机生成的字符串[%s]，%d" %(s,length))
    return s


def _generate_confusion():
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    s = ""
    for i in range(length):
        j = random.randint(0, len(confusion_set) - 1)
        s += confusion_set[j]
    return s


# 从语料库中随机获取文本行
def _get_random_text_from_corpus(corpus):
    # corpus:文本行列表
    i = random.randrange(len(corpus))
    sentence = corpus[i].strip()
    return sentence, len(sentence)


# 只在头尾加入空格
def _generate_blanks_only_head_tail(chars):
    # 随机前后加上一些空格
    _blank_num1 = random.randint(1, MAX_BLANK_NUM)
    _blank_num2 = random.randint(1, MAX_BLANK_NUM)
    return (" " * _blank_num1) + chars + (" " * _blank_num2)


# 随机在前后或者中间加入空格
def _generate_blanks_at_random_pos(chars):
    # print("%s:%d" % (chars,len(chars)))
    if not _random_accept(POSSIBILITY_BLANK): return chars
    _blank_num = random.randint(1, MAX_BLANK_NUM)
    for i in range(_blank_num):
        max_pos = len(chars)
        rand_pos = random.randint(0, max_pos)
        chars = chars[:rand_pos] + " " + chars[rand_pos:]
    # print("%s:%d" % (chars, len(chars)))
    return _generate_blanks_only_head_tail(chars)


# 从文字库中随机选择n个字符
def _get_random_text(charset):
    # 产生随机数字
    if _random_accept(POSSIBILITY_PURE_NUM):
        s_num = _generate_num()
        s = _generate_blanks_only_head_tail(s_num)
        return s, len(s)

    # 产生随机日期
    if _random_accept(POSSIBILITY_DATE):
        s_date = _generate_date()
        s = _generate_blanks_only_head_tail(s_date)
        return s, len(s)

    # 产生一些英文，因为26个字母在几千个字库中比例太小，所以必须要加强
    if _random_accept(POSSIBILITY_PURE_ENG):
        s_eng = _generator_english(charset)
        s = _generate_blanks_at_random_pos(s_eng)
        return s, len(s)

    # 对常用字及易混字增加一些样本；在测试过程中发现某些字符识别较差，可通过维护此方法获得更多样本
    if _random_accept(POSSIBILITY_CONFUSION):
        s_cf = _generate_confusion()
        s = _generate_blanks_at_random_pos(s_cf)
        return s, len(s)

    # 生成文本
    chars = _generate_words(charset)
    # 随机插入特殊字符
    chars = enhance_special_charactors(chars)  # 看看有无必要插入一些特殊需要增强的字符
    # 随机插入空格
    s = _generate_blanks_at_random_pos(chars)
    return s, len(s)


# 对一些特殊字符做多一些样本
def enhance_special_charactors(s):
    if not _random_accept(POSSIBILITY_SPECIAL): return s

    # logger.debug("原字符：%s",s)
    # specials = "|,.。-+/＊()"
    specials = ",.。、:;“”‘’()=-+"  # 常出现的符号错误
    num = random.randint(1, MAX_SPECIAL_NUM)
    for i in range(num):
        c = random.choice(specials)
        pos = random.randint(0, len(s))
        s = s[:pos] + c + s[pos:]
    # logger.debug("插入特殊字符后：%s", s)
    return s
