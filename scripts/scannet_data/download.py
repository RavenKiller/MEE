from multiprocessing import Process
import os
import wget


def download(ids):
    for id in ids:
        url = "http://kaldir.vc.in.tum.de/scannet/v2/scans/{}/{}.sens".format(id, id)
        outpath = "/root/autodl-tmp/scannet/scans/val/{}/{}.sens".format(id, id)
        print(url + " >> " + outpath)
        if os.path.exists(outpath):
            continue
        os.makedirs("/root/autodl-tmp/scannet/scans/val/{}/".format(id), exist_ok=True)
        wget.download(url, out=outpath)


if __name__ == "__main__":
    with open("/root/EvoEnc/scripts/scannet_data/scannet_scansid_test.txt", "r") as f:
        id_list = [v.replace("\n", "") for v in f.readlines()]
    N = 8
    p_list = []
    for i in range(N):
        p = Process(target=download, args=(id_list[i::N],))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    # i = 0
    # download(id_list[i::N])
