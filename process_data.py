import numpy as np
import struct
import sys
import matplotlib.pyplot as plt

def ibm2ieee2(ibm_float):
    """
    ibm2ieee2(ibm_float)
    Used by permission
    (C) Secchi Angelo
    with thanks to Howard Lightstone and Anton Vredegoor. 
    """
    dividend=float(16**6)
    
    if ibm_float == 0:
        return 0.0
    istic,a,b,c=struct.unpack('>BBBB',ibm_float)
    if istic >= 128:
        sign= -1.0
        istic = istic - 128
    else:
        sign = 1.0
    mant= float(a<<16) + float(b<<8) +float(c)
    return sign* 16**(istic-64)*(mant/dividend)
def convert2int(ibm_data):
    unpack_data = struct.unpack('i',ibm_data)
    d3 = (unpack_data[0] << 24) & 0xff000000
    d2 = (unpack_data[0] << 8) & 0x00ff0000
    d1 = (unpack_data[0] >> 8) & 0x0000ff00
    d0 = (unpack_data[0] >> 24) & 0x000000ff
    val = d3 | d2 | d1 | d0
    return val
def get_data(file_name, hrz, sample_num, start_inline, start_xline, min_inline, max_inline, min_xline, max_xline, up_points, down_points, time_delay):
    print(file_name, 'begins')
    points_num = down_points - up_points
    pre_data = np.zeros([(max_inline-min_inline+1),(max_xline-min_xline+1),points_num])
    fp = open(file_name,'rb')
    fp.seek(3600,0)
    status = 0
    count = 0
    trace_num = (max_inline-min_inline+1)*(max_xline-min_xline+1)
    while(fp):
        fp.seek(188,1)
        ibm_data = fp.read(4)
        if not ibm_data:
            break
        inline = convert2int(ibm_data)
        ibm_data = fp.read(4)
        xline = convert2int(ibm_data)
        fp.seek(44,1)
        if (inline>=min_inline) and (inline<=max_inline) and (xline>=min_xline) and (xline<=max_xline):
            count = count + 1
            time = int(hrz[(inline-start_inline),(xline-start_xline)]) - time_delay
            fp.seek((time+up_points)*4,1)
            data = fp.read(points_num*4)
            fp.seek((sample_num-time-up_points-points_num)*4,1)
            trace_data = []
            for k in range(points_num):
                index_begin = k*4
                index_end = (k+1)*4
                trace_data.append(ibm2ieee2(data[index_begin:index_end]))
            pre_data[(inline-min_inline),(xline-min_xline),:] = trace_data
            if count == (status+int(trace_num*0.1)):
                status = status+int(trace_num*0.1)
                print(file_name, round(status*100/trace_num), '%')
            #plt.plot(trace_data)
            #plt.show()
        else:
            fp.seek(sample_num*4,1)
    fp.close()
    print(file_name, '100% completed')
    return pre_data
if __name__ == '__main__':
    start_inline = 284
    end_inline = 1427
    start_xline = 206
    end_xline = 950
    num_inline = end_inline - start_inline + 1
    num_xline = end_xline - start_xline + 1
    min_inline = 351
    max_inline = 1300
    min_xline = 301
    max_xline = 850
    sample_num = 5001
    time_delay = 0
    
    hrz = np.zeros([num_inline,num_xline])
    
    for line in open('hrz/P2l.dat'):
        line_arr = line.strip().split()
        cur_inline = int(line_arr[0])
        cur_xline = int(line_arr[1])
        if cur_inline<start_inline or cur_inline>end_inline or cur_xline<start_xline or cur_xline>end_xline:
            continue
        hrz[(cur_inline-start_inline),(cur_xline-start_xline)] = round(float(line_arr[2]))
    
    files = []
    files.append('2012LL3D_PSTMDSP_AZIM_000_030.SEGY')
    files.append('2012LL3D_PSTMDSP_AZIM_030_060.SEGY')
    files.append('2012LL3D_PSTMDSP_AZIM_060_090.SEGY')
    files.append('2012LL3D_PSTMDSP_AZIM_090_120.SEGY')
    files.append('2012LL3D_PSTMDSP_AZIM_120_150.SEGY')
    files.append('2012LL3D_PSTMDSP_AZIM_150_180.SEGY')
    file_num = 6
    up_points = -11
    down_points = 13
    points_num = down_points - up_points
    single_data = []
    for i in range(file_num):
        data = get_data(files[i] ,hrz, sample_num, start_inline, start_xline, min_inline, max_inline, min_xline, max_xline, up_points, down_points, time_delay)
        single_data.append(data)

    print('all data have been processed')

    processed_data = np.zeros([(max_inline-min_inline+1),(max_xline-min_xline+1),(points_num*file_num)])
    for i in range(max_inline-min_inline+1):
        for j in range(max_xline-min_xline+1):
            for k in range(file_num):
                processed_data[i,j,(points_num*k):(points_num*(k+1))]    = single_data[k][i,j,:]
            
    print('merging is done')
    np.save('6positions_24points.npy', processed_data)
    
    #post_data = get_data('PSTM_STK_ALL_AGC.segy', hrz, 2001, min_inline, max_inline, min_xline, max_xline)
    #np.save('post_data.npy', post_data)
    
    
    
