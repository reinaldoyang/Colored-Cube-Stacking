import socket
import time
HOST = '192.168.0.1' # arm ip
PORT = 3000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
time.sleep(5)


# s.bind(('', PORT))
# # display client address
# print("CONNECTION FROM:", str(addr))
# # s.send(bytes("hola", "utf-8"))

while True:
    s.send(bytes("hola", "utf-8"))
    time.sleep(5)