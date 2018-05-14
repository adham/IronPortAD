"""
extract MID lines from raw data and stich them together
"""

import os
import re
import codecs
import numpy as np
from tqdm import tqdm


def read_logfile(file_name):
    """Read a og file
    """
    with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as fp:
        data = fp.readlines()
        data = [dd.strip() for dd in data]
    return data


def find_MID(line):
    pattern = r'MID [0-9]*'
    match = re.search(pattern, line)
    s = match.group(0)
    mid = s.split()[1]
    return mid


def find_MID_lines(line_number, data):
    MID = find_MID(data[line_number])

    tmp = 'MID {}'.format(MID)
    idxs = [tmp in dd for dd in data]
    MID_lines = [data[i] for i, idx in enumerate(idxs) if idx]
    return MID_lines


def write_to_file(user, lines):
    file_name = user.lower().split('@')[0] + '.txt'
    with open(file_name, 'a') as fp:
        for line in lines:
            fp.write('{}\n'.format(line))
        fp.write('\n\n\n')


users = """
Tyronne.Anderson@team.telstra.com
Revathi.Pisharody@team.telstra.com
Manikandan.Ramaraj@team.telstra.com
Danny.S.Chea@team.telstra.com
Cass.Rowell@team.telstra.com
Manjula.S@team.telstra.com
Jenni.Dean@team.telstra.com
Scott.Zwanenbeek@team.telstra.com
Aaron.Leung@team.telstra.com
Tarnya.Dunning@team.telstra.com""".split()

dir_path = '/mnt/telstra/Data_2018/IronPort-ESA/2017/07'
file_names = [os.path.join(dir_path, f) for dir_path, dir_names, files in os.walk(dir_path) \
    for f in files if not f.endswith('.s')]

for file_name in tqdm(file_names, desc='files'):
    data = read_logfile(file_name)
    for user in tqdm(users, desc='users'):
        idxs = [user in dd for dd in data]
        idxs = np.where(idxs)[0]
        for idx in tqdm(idxs, desc='write'):
            write_to_file(user, find_MID_lines(idx, data))
