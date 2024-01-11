# From https://github.com/lukas-blecher/LaTeX-OCR

import os
import re
import sys
import shutil
import logging
import argparse
import subprocess

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s\t%(levelname)s\t%(name)s %(filename)s:%(lineno)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Temporary files
UNCLEAN_FILE_PATH = os.path.join(os.path.dirname(__file__), 'tmp_data/unclean.txt')
CLEAN_FILE_PATH = os.path.join(os.path.dirname(__file__), 'tmp_data/clean.txt')


def normalize_data(data):
    preprocess_to_txt(data)
    
    # Normalize data
    assert os.path.exists(UNCLEAN_FILE_PATH), UNCLEAN_FILE_PATH
    shutil.copy(UNCLEAN_FILE_PATH, CLEAN_FILE_PATH)
    operators = '\s?'.join('|'.join([
        'arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot',
        'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
        'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max',
        'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh'])
    )
    ops = re.compile(r'\\operatorname {(%s)}' % operators)
    temp_file = CLEAN_FILE_PATH + '.tmp'
    with open(temp_file, 'w') as fout:
        prepre = open(CLEAN_FILE_PATH, 'r').read().replace('\r', ' ')  # delete \r
        # replace split, align with aligned
        prepre = re.sub(
            r'\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}',
            r'\\begin{aligned}\2\\end{aligned}', prepre, flags=re.S
        )
        prepre = re.sub(
            r'\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}',
            r'\\begin{matrix}\2\\end{matrix}',
            prepre,
            flags=re.S
        )
        fout.write(prepre)

    cmd = r"cat %s | node %s %s > %s " % (
        temp_file,
        os.path.join(os.path.dirname(__file__), 'preprocess_latex.js'),
        'normalize',
        CLEAN_FILE_PATH
    )
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logger.info("FAILED: {}".format(cmd))
    temp_file = CLEAN_FILE_PATH + '.tmp'
    shutil.move(CLEAN_FILE_PATH, temp_file)
    with open(temp_file, 'r') as fin:
        with open(CLEAN_FILE_PATH, 'w') as fout:
            count_str = 0
            for line in fin:
                count_str = count_str + 1
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                if len(tokens_out) > 5:
                    post = ' '.join(tokens_out)
                    # use \sin instead of \operatorname{sin}
                    names = ['\\'+x.replace(' ', '') for x in re.findall(ops, post)]
                    post = re.sub(
                        ops,
                        lambda match: str(names.pop(0)),
                        post
                    ).replace(r'\\ \end{array}', r'\end{array}')
                    fout.write(post+'\n')
                else:
                    fout.write('ERROR!\n')
    os.remove(temp_file)

    result = []
    clean_data_file = open(CLEAN_FILE_PATH,'r')
    for value in clean_data_file:
        if value != 'ERROR!\n':
            result.append(value[:-1])
    clean_data_file.close()

    return result


def preprocess_to_txt(data):
    unclean_file = open(UNCLEAN_FILE_PATH,'w')
    for i in range(len(data)):
        tmp_str = data[i].replace('\\begin{align*}','\\begin{align}')
        tmp_str = tmp_str.replace('\begin{align*}','\\begin{align}')
        tmp_str = tmp_str.replace('\\end{align*}','\\end{align}')
        tmp_str = tmp_str.replace('\end{align*}','\\end{align}')
        tmp_str = tmp_str.replace('\\begin{gather*}','\\begin{align}')
        tmp_str = tmp_str.replace('\begin{gather*}','\\begin{align}')
        tmp_str = tmp_str.replace('\\end{gather*}','\\end{align}')
        tmp_str = tmp_str.replace('\end{gather*}','\\end{align}')
        if tmp_str.startswith('$') and tmp_str.endswith('$'):
            tmp_str = tmp_str[1:-1]
            tmp_str = '\\begin{align} ' + tmp_str + ' \\end{align}'
        unclean_file.write(tmp_str)
        unclean_file.write('\n')
    unclean_file.close()