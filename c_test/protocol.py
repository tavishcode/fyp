#!/usr/bin/env python3

import argparse

import stat
import os

# Tell Metis to cache this data packet (without cache replacement)
def reply_cache(fifo_send):
	fifo_send.write('Y')
	fifo_send.flush()

# Tell Metis no need to cache this data packet
def reply_nocache(fifo_send):
	fifo_send.write('N')
	fifo_send.flush()

# Tell Metis to cache this data packet, with "victim" being the victim of replacement
def reply_replace_cache(fifo_send, victim):
	fifo_send.write('R')
	fifo_send.write(victim)
	fifo_send.flush()

def worker(fifo_recv_path, fifo_send_path):
	if not stat.S_ISFIFO(os.stat(fifo_recv_path).st_mode):
		os.mkfifo(fifo_recv_path)

	if not stat.S_ISFIFO(os.stat(fifo_send_path).st_mode):
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

		message_type, name = data.split()

		if message_type == 'I':
			# Received interest
			print(f'Recv Interest: {name}')

		elif message_type == 'D':
			# Received data
			print(f'Recv Data: {name}')

			# No need to cache
			# reply_nocache(fifo_send)

			# Cache without replacement
			# reply_cache(fifo_send)

			# Cache with replacement (victim: ccnx:/serverA/123)
			reply_replace_cache(fifo_send, 'ccnx:/serverA/123')

	fifo_recv.close()
	fifo_send.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('capacity', type=str, help='capacity of content store') 
	parser.add_argument('fifo_recv', type=str, help='path to read FIFO')
	parser.add_argument('fifo_send', type=str, help='path to write FIFO')

	args = parser.parse_args()

	worker(args.fifo_recv, args.fifo_send)

if __name__ == '__main__':
	main()