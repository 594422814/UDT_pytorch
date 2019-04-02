from os.path import join, isdir
from os import listdir
import argparse
import json
import glob
import xml.etree.ElementTree as ET
import pdb

parser = argparse.ArgumentParser(description='Parse the VID Annotations for training DCFNet')
parser.add_argument('data', metavar='DIR', help='path to VID')
args = parser.parse_args()

print('VID2015 Data:')
VID_base_path = args.data
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
img_base_path = join(VID_base_path, 'Data/VID/train/')
sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
# sub_sets = sorted({'ILSVRC2015_VID_train_0000', 'ILSVRC2015_VID_train_0001', 'ILSVRC2015_VID_train_0002', 'ILSVRC2015_VID_train_0003'})
total_frame = 0

vid = []
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)
    videos = sorted(listdir(sub_set_base_path))
    s = []
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
        v = dict()
        v['base_path'] = join(img_base_path, sub_set, video)
        v['frame'] = []
        video_base_path = join(sub_set_base_path, video)
        xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
        for xml in xmls:
            total_frame += 1
            f = dict()
            xmltree = ET.parse(xml)
            size = xmltree.findall('size')[0]
            frame_sz = [int(it.text) for it in size]
            f['frame_sz'] = frame_sz
            f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
            v['frame'].append(f)
        s.append(v)
    vid.append(s)
print('save json (raw vid info), please wait 1 min~')
print('total frame number: {:d}'.format(total_frame))
json.dump(vid, open('vid.json', 'w'), indent=2)
print('done!')

