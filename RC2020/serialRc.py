# -*- coding: utf-8 -*-
import time
import serial
import struct

class vSerial:
    def __init__(self, baudrate: int = 9600):
        self.serial_port = serial.Serial(
            port="/dev/ttyTHS0",
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.data = 0
        time.sleep(1)

    # 发送一个字节
    # senders为16进制字符串，例如 senders = 'FF' 表示发送0xFF
    # 必须为两位，1写成'01'

    def sendOneByte(self, bytes_mesg: bytes):
        self.serial_port.write(bytes_mesg)

    def sendOneByteInHex(self, senders):
        try:
            self.serial_port.write(bytes.fromhex(senders))
        except Exception as exception_error:
            print("Error occurred.")
            print("Error: " + str(exception_error))

    def sendOneByteInIntEx(self, send_int: int, bits: int = 8):
        """
        接收10进制数据并发送

        参数
        -----
        send_int - int
            待发送的整数数据，如果超过指定位的最大值会限制在最大值大小
        bits - int
            指定发送数据的位数
        """
        send_int = int(send_int)
        maxval = 255 if bits == 8 else 65535
        flag = '>B' if bits == 8 else '>H'
        if send_int > maxval:
            send_int = maxval
        elif send_int < 0:
            send_int = 0
        try:
            send_int = struct.pack(flag, send_int)
            self.serial_port.write(send_int)
        except Exception as exception_error:
            print('Error occurred.')
            print('Error: ', exception_error)
            

    # 发送一个字节
    # senders为10进制int整型，范围在0-255之间，例如 senders = 255 表示发送0xFF
    def sendOneByteInInt(self, senders, bits: int = 8):
        maxval = 255
        if senders > maxval:
            senders = maxval
        if senders < 0:
            senders = 0
        try:
            if senders <= 15:
                senders += 16
                self.serial_port.write(bytes.fromhex(hex(senders).replace('0x1', '0')))
            else:
                self.serial_port.write(bytes.fromhex(hex(senders).replace('0x', '')))
        except Exception as exception_error:
            print("Error occurred.")
            print("Error: " + str(exception_error))

    def readData(self, read_bytes: int = 128):
        "若无数据返回上一次的self.data，有数据返回int类型"
        if self.serial_port.inWaiting() > 0:
            data = self.serial_port.read()
            data = data.decode()
            data = data.encode("unicode_escape")
            data = data.decode("ascii")
            self.data = int(data.replace("\\x",""))
        return self.data

if __name__ == '__main__':
    vas = vSerial()
    numbers = 0
    while True:
        numbers = vas.readData()
        print(numbers)