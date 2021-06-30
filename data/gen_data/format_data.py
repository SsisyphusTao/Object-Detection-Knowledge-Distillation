from PIL import Image
import json
from os import listdir
import re
import numpy as np
import shutil
totals = []
count = 0
converter = { 'person': 0, 'hat': 2, 'helmet': 2, 'head': 1}
for n in range(1, 13):
  with open('/home/chandler/datasets/pedestrian/S3-5000-worker-helmet/json/%i.json'%n, 'r') as f:
      j = json.load(f)

  for s in j['samples']:
    path = "/home/chandler/datasets/pedestrian/S3-5000-worker-helmet/images/%i/%s"%(n, s['imageUrl'].split('/')[-1])
    totals.append({'imgpath':path, 'bboxes':[], 'size':(1920, 1080)})
    dest = '/home/chandler/datasets/together/images/train/%s' % s['imageUrl'].split('/')[-1]
    shutil.copy(path, dest)
    count+=1
    with open('/home/chandler/datasets/together/labels/train/%i-%s' % (n,s['imageUrl'].split('/')[-1].replace('.jpg', '.txt')), 'a+') as f:
      for i in s['annotation']:
        try:
          if i['classification'] == 'person' or i['classification']=='head' or i['classification']=='helmet':
            x = i['centerX']
            y = i['centerY']
            w = i['width']
            h = i['height']

            l = float((x-w/2))
            u = float((y-h/2))
            r = float((x+w/2))
            d = float((y+h/2))
            # totals[-1]['bboxes'].append((l,u,r,d))
            f.write('%d %f %f %f %f\n' % (converter[i['classification']], l, u, r, d))
        except:
          pass
print(count)

with open('/home/chandler/datasets/pedestrian/WiderPerson/train.txt', 'r') as f:
  a = f.readlines()

for i in a:
  i = i[:-1]
  path = '/home/chandler/datasets/pedestrian/WiderPerson/Images/%s.jpg'%i
  size = Image.open(path).size
  totals.append({'imgpath':path, 'bboxes':[], 'size':(size[0], size[1])})
  dest = '/home/chandler/datasets/together/images/train/wider-%s.jpg' % i
  shutil.copy(path, dest)
  count+=1
  with open('/home/chandler/datasets/pedestrian/WiderPerson/Annotations/%s.jpg.txt'%i) as f:
    anns = f.readlines()
  with open('/home/chandler/datasets/together/labels/train/wider-%s.txt' % i, 'a+') as f:
    for j in anns[1:]:
      j = j.split(' ')
      if j[0]=='1' or j[0] =='3' or j[0] =='2':
        # totals[-1]['bboxes'].append((int(j[1]), int(j[2]), int(j[3]), int(j[4])))
        f.write('%d %f %f %f %f\n' % (0, float(j[1])/size[0], float(j[2])/size[1], float(j[3])/size[0], float(j[4])/size[1]))
print(count)


with open('/home/chandler/datasets/pedestrian/WiderPerson/val.txt', 'r') as f:
  a = f.readlines()

for i in a:
  i = i[:-1]
  path = '/home/chandler/datasets/pedestrian/WiderPerson/Images/%s.jpg'%i
  size = Image.open(path).size
  totals.append({'imgpath':path, 'bboxes':[], 'size':(size[0], size[1])})
  dest = '/home/chandler/datasets/together/images/val/wider-%s.jpg' % i
  shutil.copy(path, dest)
  count+=1
  with open('/home/chandler/datasets/pedestrian/WiderPerson/Annotations/%s.jpg.txt'%i) as f:
    anns = f.readlines()
  with open('/home/chandler/datasets/together/labels/val/wider-%s.txt' % i, 'a+') as f:
    for j in anns[1:]:
      j = j.split(' ')
      if j[0]=='1' or j[0] =='3' or j[0] =='2':
        # totals[-1]['bboxes'].append((int(j[1]), int(j[2]), int(j[3]), int(j[4])))
        f.write('%d %f %f %f %f\n' % (0, float(j[1])/size[0], float(j[2])/size[1], float(j[3])/size[0], float(j[4])/size[1]))
print(count)

# with open('/home/chandler/datasets/pedestrian/person_bbox_train.json', 'r') as f:
#   j = json.load(f)

# for i in j.keys():
#   path = '/home/chandler/datasets/pedestrian/'+i
#   size = Image.open(path).size
#   totals.append({'imgpath':path, 'bboxes':[], 'size':(size[0], size[1])})
#   count+=1
#   for z in j[i]["objects list"]:
#     if z['category'] == 'person':
#       x0 = int(z['rects']["full body"]['tl']['x'] * 1920)
#       y0 = int(z['rects']["full body"]['tl']['y'] * 1080)
#       x1 = int(z['rects']["full body"]['br']['x'] * 1920)
#       y1 = int(z['rects']["full body"]['br']['y'] * 1080)
#       totals[-1]['bboxes'].append((x0, y0, x1, y1))
# print(count)
# for i in listdir('/home/chandler/datasets/pedestrian/PennFudanPed/Annotation'):
#   with open('/home/chandler/datasets/pedestrian/PennFudanPed/Annotation/'+i, 'r') as f:
#     a = f.readlines()
#   for j in a:
#     if 'Image filename' in j:
#       path = '/home/chandler/datasets/pedestrian/'+j.split('"')[-2]
#       size = Image.open(path).size
#       totals.append({'imgpath':path, 'bboxes':[], 'size':(size[0], size[1])})
#       count+=1
#     if 'Xmin, Ymin' in j:
#       r = re.compile(r'\d+')
#       [_, x0, y0, x1, y1] = r.findall(j)
#       totals[-1]['bboxes'].append((int(x0), int(y0), int(x1), int(y1)))
# print(count)

# with open('/home/chandler/datasets/brainwash/brainwash_train.idl', 'r') as f:
#     train_list = f.readlines()
# for i in train_list:
#     [a] = json.loads(('[{%s}]'%i.replace('";\n', '":[]').replace(': ', ':[').replace(';\n', ']').replace('(', '[').replace(')', ']').replace('.\n', ']')).replace('[{"', '[{"/home/chandler/datasets/brainwash/'))
#     path = list(a.keys())[0]
#     totals.append({'imgpath':path, 'bboxes':[], 'size':(640, 480)})
#     for z in a[path]:
#       totals[-1]['bboxes'].append((int(z[0]),int(z[1]), int(z[2]),int(z[3])))
  
# with open('/home/chandler/Night_Detection/trainset.json', 'w') as f:
#   json.dump(totals, f)