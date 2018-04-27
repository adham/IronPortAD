"""
This script parses IronPort ESA logs into sender_data and recipient_data.

Deakin University,
PRaDA - Adham Beyki, odinay@gmail.com

NOTE:
Multiprocessing did not increase the speed that much to be worthy if sacrifying multiple CPUs.
Better to use one CPU but launch many tasks simultaneasly.
"""

import os
import re
import codecs
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def read_logfile(file_name):
    """Read a og file
    """
    with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as fp:
        data = fp.readlines()
        data = [dd.strip() for dd in data]

    # fidn time offset of the logfile
    # line = data[2]
    # line = line.split()
    # idx = line.index('seconds')
    # time_offset = int(line[idx-1])

    return data


def find_from_lines(data):
    """Finds lines associated with senders
    """
    lines = [line for line in data if 'From: <' in line]

    # remove `From: <>`
    ret = [line for line in lines if '<>' not in line]

    return ret


def find_email(line):
    """Find email address in the line
    """
    try:
        aa = line.find('<')
        bb = line.find('>')
        ret = line[aa+1:bb]
    except:
        ret = None

    return ret


def find_time(line):
    return pd.to_datetime(line[:24])


def find_MID(line):
    pattern = r'MID [0-9]*'
    match = re.search(pattern, line)
    s = match.group(0)
    mid = s.split()[1]
    return mid


def find_to(line, log_data):
    """Fine recipients of a from entry
    """
    out = None
    mid = find_MID(line)
    lines = [l for l in log_data if 'To: <' in l and mid in l]

    if len(lines)>0:
        out = [find_email(l) for l in lines]

    return out


def process_line(line, log_data):

    out = None
    sender = find_email(line)
    if sender is not None:

        # 1. it's either sent from a telstra email
        if 'telstra' in sender:
            tos = find_to(line, log_data)
            try:
                tt = find_time(line)
                out = ['sent', [sender, tt, tos]]
            except:
                pass

        # 2. or recived by a telstra email
        else:
            tos = find_to(line, log_data)
            try:
                tt = find_time(line)
                out = ['received', [tos, tt, sender]]
            except:
                pass
    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--nlines', type=int, default=30000, help='number of lines to read per chunk')
    parser.add_argument('--date', type=str, help='date to parse')
    parser.add_argument('--logs', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    # get file name associated with date passed
    print()
    print('===> finding files associated with {}'.format(args.date))
    yy, mm, dd = args.date.split('-')
    log_dir = os.path.join(args.logs, yy, mm, dd)
    file_names = [os.path.join(dir_path, f) for dir_path, dir_names, files in os.walk(log_dir) \
        for f in files if not f.endswith('.s')]
    print()
    print('{} files found'.format(len(file_names)))

    # init data containers
    sender_data = defaultdict(list)
    recipient_data = defaultdict(list)

    # loop through file names
    print()
    print('===> parsing')
    for file_name in tqdm(file_names, desc='files '):
        log_data = read_logfile(file_name)
        file_len = int((subprocess.Popen('wc -l {}'.format(file_name),
            shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])
        nb_chunks = file_len//args.nlines

        residue = []
        for chunk_i in tqdm(range(nb_chunks), desc='chunks'):
            chunk_start = chunk_i*args.nlines
            chunk_end = chunk_start+args.nlines
            if chunk_i == nb_chunks-1:
                chunk_end = file_len-1

            log_data_chunk = log_data[chunk_start:chunk_end]

            # find lines containing `From`
            from_lines = find_from_lines(log_data_chunk)

            # process lines
            data = []
            for line in tqdm(from_lines, desc='lines '):
                result = process_line(line, log_data_chunk)
                if (result is None) or (result[1][0] is None):
                    residue.append(line)
                else:
                    data.append(result)

            # write data to sender and recipient containers
            for dd in data:
                if dd[0] == 'sent':
                    sender_data[dd[1][0]].append(dd[1][1:])
                else:
                    for to in dd[1][0]:
                        recipient_data[to].append(dd[1][1:])

        # parse residues
        data = []
        for line in residue:
            result = process_line(line, log_data)
            if (result is not None) and (result[1][0] is not None):
                data.append(result)
        for dd in data:
            if dd[0] == 'sent':
                sender_data[dd[1][0]].append(dd[1][1:])
            else:
                for to in dd[1][0]:
                    recipient_data[to].append(dd[1][1:])


    # write data containers to disck
    pd.to_pickle(sender_data, os.path.join(args.out, 'sender_data_{}.pkl'.format(args.date)))
    pd.to_pickle(recipient_data, os.path.join(args.out, 'recipient_data_{}.pkl'.format(args.date)))


if __name__ == '__main__':
    main()
