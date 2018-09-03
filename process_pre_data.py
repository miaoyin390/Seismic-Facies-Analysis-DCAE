import numpy as np
import struct
import sys

def convert2int(ibm_data):
    unpack_data = struct.unpack('i',ibm_data)
    d3 = (unpack_data[0] << 24) & 0xff000000
    d2 = (unpack_data[0] << 8) & 0x00ff0000
    d1 = (unpack_data[0] >> 8) & 0x0000ff00
    d0 = (unpack_data[0] >> 24) & 0x000000ff
    val = d3 | d2 | d1 | d0
    return val

if __name__ == '__main__':
    file_name = '/media/yinm/SED/cmp_nmo_gather/cmp_nmo_all.sgy'
    sample_num = 1201
    fp = open(file_name, 'rb')
    fp.seek(3600,0)
    for i in range(1000):
        fp.seek(180,1)
        ibm_data = fp.read(4)
        inline = convert2int(ibm_data)
        ibm_data = fp.read(4)
        xline = convert2int(ibm_data)
        fp.seek(52,1)
        fp.seek((sample_num*4), 1)
        print(inline, xline)
