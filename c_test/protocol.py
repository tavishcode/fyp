#!/usr/bin/env python3

import argparse
import stat
import os
import signal

# cache imports
from contentstore import PretrainedCNNContentStore as cs

# Tell Metis to cache this data packet (without cache replacement)


def reply_cache(fifo_send):
    fifo_send.write('Y')
    fifo_send.flush()

# Tell Metis no need to cache this data packet


def reply_nocache(fifo_send):
    fifo_send.write('N')
    fifo_send.flush()

# Tell Metis to cache this data packet, with "victim" being the victim of
# replacement


def reply_replace_cache(fifo_send, victim_server, victim_page):
    fifo_send.write('R')
    fifo_send.write(f'{victim_server} {victim_page}')
    fifo_send.flush()


def worker(capacity, fifo_recv_path, fifo_send_path):
    cache = cs(capacity)

    def update_rankings(signal, frame):
        print('Update rankings')
        cache.update_rankings_wrapper()

    signal.signal(signal.SIGUSR1, update_rankings)

    try:
        if not stat.S_ISFIFO(os.stat(fifo_recv_path).st_mode):
            os.mkfifo(fifo_recv_path)
    except FileNotFoundError:
        os.mkfifo(fifo_recv_path)

    try:
        if not stat.S_ISFIFO(os.stat(fifo_send_path).st_mode):
            os.mkfifo(fifo_send_path)
    except FileNotFoundError:
        os.mkfifo(fifo_send_path)

    print('Opening FIFO')
    fifo_recv = open(fifo_recv_path, 'r')
    fifo_send = open(fifo_send_path, 'w')
    print('FIFO opened')

    while True:
        data = fifo_recv.readline()
        if len(data) == 0:
            print("Other side closed")
            break

        # Metis received message
        # message_type: {I, D}. I = interest D = data
        # server: server name
        # page: page index of wiki dataset. Index starts at 1.
        message_type, server, page, day = data.split()

        day = int(day)
        page = int(page)

        if message_type == 'I':
            # Received interest

            # Records statistics
            cache.update_stats(day, (server, page))
            cache.get((server, page))

        elif message_type == 'D':
            # Received data

            should_cache, victim = cache.add((server, page))

            if should_cache:  # Cache content ?
                if victim == None:  # Cache without replacement ?
                    reply_cache(fifo_send)
                else:
                    victim_server, victim_page = victim
                    reply_replace_cache(fifo_send, victim_server, victim_page)
            else:
                reply_nocache(fifo_send)

    fifo_recv.close()
    fifo_send.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capacity', type=int, help='capacity of content store')
    parser.add_argument('fifo_recv', type=str, help='path to read FIFO')
    parser.add_argument('fifo_send', type=str, help='path to write FIFO')
    args = parser.parse_args()

    worker(args.capacity, args.fifo_recv, args.fifo_send)


if __name__ == '__main__':
    main()