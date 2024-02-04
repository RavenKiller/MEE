import argparse
import os
import sys
from multiprocessing import Process
import time

from SensorData import SensorData

filename = ""


def extract(ids):
    # print(ids)
    for id in ids:
        filename = "/root/autodl-tmp/scannet/scans/train/{}/{}.sens".format(id, id)
        output_path = "/root/autodl-tmp/data/stage1/scannet-depth/train"
        if not os.path.exists(filename):
            print("Skip "+filename)
            continue
        print("Extract "+filename)
        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError:
            pass
        failed = False
        try:
            sd = SensorData(filename)
            sd.export_depth_images(os.path.join(output_path, "depth"),id)
        except:
            failed = True
        if not failed:
            print("Remove "+filename)
            os.remove(filename)


if __name__ == "__main__":
    with open("/root/EvoEnc/scripts/scannet_data/scannet_scansid.txt", "r") as f:
        id_list = [v.replace("\n", "") for v in f.readlines()]
    N = 1
    p_list = []
    for i in range(N):
        p = Process(target=extract, args=(id_list[i::N],))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()