from os import listdir, path
import xml.dom.minidom
converter = { 'person': 1, 'hat': 2, 'helmet': 2, 'head': 1}
def parseXml(path):
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement

    objects = collection.getElementsByTagName("object")
    [size] = collection.getElementsByTagName("size")
    width = float(size.getElementsByTagName("width")[0].childNodes[0].nodeValue)
    height = float(size.getElementsByTagName("height")[0].childNodes[0].nodeValue)
    bboxes = []
    labels = []
    for i in objects:
        label = i.getElementsByTagName('name')[0].childNodes[0].nodeValue
        if label in converter.keys():
            labels.append(converter[label])
        else:
            continue
        bndbox = i.getElementsByTagName('bndbox')[0]
        l = float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue) / width
        t = float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue) / height
        r = float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue) / width
        d = float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue) / height
        bboxes.append([l,t,r,d])
    return bboxes, labels

for x in listdir('/home/chandler/datasets/openhelmet/annotations'):
    bboxes, labels = parseXml(path.join('/home/chandler/datasets/openhelmet/annotations', x))
    with open(path.join('/home/chandler/datasets/together/labels/train', x.replace('.xml', '.txt')), "a+") as f:
        for n, i in enumerate(bboxes):
            f.write('%d %f %f %f %f\n' % (labels[n], i[0], i[1], i[2], i[3]))