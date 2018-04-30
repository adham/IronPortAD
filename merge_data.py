import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def find_files(pin, tag):

    file_names = os.listdir(pin)
    file_names = [ff for ff in file_names if tag in ff]
    file_names = [os.path.join(pin, ff) for ff in file_names]

    return file_names


def main(pin, pout):

    sender_files = find_files(pin, 'sender')
    recipient_files = find_files(pin, 'recipient')

    sender_data = defaultdict(list)
    recipient_data = defaultdict(list)

    print('===> building senders\' dictionary')
    for file_name in tqdm(sender_files):
        data = pd.read_pickle(file_name)
        for k, v in data.items():
            sender_data[k] += v

    print()
    print('===> building recipients\' dictionary')
    for file_name in tqdm(recipient_files):
        data = pd.read_pickle(file_name)
        for k, v in data.items():
            recipient_data[k] += v


    print()
    print('===> writing the redult on disk')
    pd.to_pickle({'sender_data': sender_data, 'recipient_data': recipient_data}, pout)


if __name__ == '__main__':
    args = sys.argv

    pin = args[1]
    pout = args[2]
    main(pin, pout)
