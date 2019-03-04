class Event:
    def __init__(self, actor_name, time, func, pkt, src):
        self.time = time
        self.func = func
        self.pkt = pkt
        self.src = src
        self.actor_name = actor_name
        self.next = None
        self.prev = None

class EventList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    def add(self, event):
        if self.head == None:
            self.head = event
            self.tail = event
        else:
            if event.time <= self.head.time:
                self.head.prev = event
                event.next = self.head
                self.head = event
            elif event.time >= self.tail.time:
                self.tail.next = event
                event.prev = self.tail
                self.tail = self.tail.next
            else:
                curr = self.head.next
                while curr.time < event.time:
                    curr = curr.next
                temp = curr.prev
                curr.prev = event
                event.next = curr
                if temp != None:
                    event.prev = temp
                    temp.next = event
        self.length += 1

    def popfront(self):
        event = self.head
        if self.head != None:
            if self.length == 1:
                self.head = None
                self.tail = None
            else:
                self.head = self.head.next
                self.head.prev.next = None
                self.head.prev = None
            self.length -= 1
            return event
        else:
            return None

    def print_list(self):
        curr = self.head
        print('start list')
        while curr != None:
            print(curr.time)
            curr = curr.next
        print('end list')

    def peek(self):
        if self.head != None:
            return self.head.actor_name, self.head.time
        else:
            return None, None

# a = EventList()

# a.add(Event(1,'a','b',None))
# a.add(Event(0,'a','b',None))
# a.add(Event(2,'a','b',None))
# a.popfront()
# a.add(Event(-1,'a','b',None))

# curr = a.head
# while curr!=None:
#     print(curr.time)
#     curr = curr.next
