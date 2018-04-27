import os
import sys

def find_files(pin, tag):

    file_names = os.listdir(pin)
    file_names = [ff for ff in file_names if tag in ff]
    1/0

def main(pin):

    sender_files = find_files(pin, 'sender')
    pass


if __name__ == '__main__':
    args = sys.argv()

    data_dir = args[1]
    main(data_dir)
