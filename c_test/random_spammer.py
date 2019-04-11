import random

content_types = 200
reqs_per_day = 10000
routers = 10
days = 10

def refresh_all():
    for router in range(routers):
        path = "/tmp/fifo%d" % router
        fifo = open(path,'w')
        fifo.write('refresh\n')
        fifo.close()
        fifo = open(path, "r")
        info =  fifo.read()
        fifo.close()

for day in range(days):
    for req in range(reqs_per_day):
        path = "/tmp/fifo%d" % random.randint(0,9)
        fifo = open(path,'w')
        fifo.write('interest %d\n' % (random.randint(0,content_types-1)))
        fifo.close()
        fifo = open(path, "r")
        info =  fifo.read()
        fifo.close()
    refresh_all()

