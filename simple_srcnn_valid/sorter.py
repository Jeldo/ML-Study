import os
import sys
import re


# 0~63 folders to feature_mapure folders

def sorter(mode):
    base_path = os.getcwd()

    # feature-wise to layer-wise
    if mode == 0:
        print('sort by feature-wise to layer-wise')
        first = re.compile(r'1st[^ ]+')
        second = re.compile(r'2nd[^ ]+')
        third = re.compile(r'3rd[^ ]+')

        if not os.path.exists(base_path+'/layer'):
            os.mkdir(base_path+'/layer')
            os.mkdir(base_path+'/layer/1')
            os.mkdir(base_path+'/layer/2')
            os.mkdir(base_path+'/layer/3')

        for i in range(0, 64):
            img_path = base_path + '/feature_map/{}'.format(i)
            for img in os.listdir(img_path):
                ff = re.match(first, img)
                if ff != None:
                    os.rename(img_path+'/'+ff.group(), base_path +
                              '/layer/1/'+ff.group())
                ss = re.match(second, img)
                if ss != None:
                    os.rename(img_path+'/'+ss.group(), base_path +
                              '/layer/2/'+ss.group())
                tt = re.match(third, img)
                if tt != None:
                    os.rename(img_path+'/'+tt.group(), base_path +
                              '/layer/3/'+tt.group())
    # layer-wise to feature-wise
    else:
        print('sort by layer-wise to feature-wise')
        for i in range(0, 3):
            img_path = base_path + '/layer/{}/'.format(i+1)
            for img in os.listdir(img_path):
                for j in range(0, 64):
                    pattern_name = re.compile(r'[^ ]*_{}[.]png'.format(j))
                    img_name = re.match(pattern_name, img)
                    if img_name != None:
                        img_number = str(j)
                        if not os.path.exists(base_path+'/feature_map'):
                            os.mkdir(base_path+'/feature_map')
                        if not os.path.exists(base_path+'/feature_map/{}'.format(j)):
                            os.mkdir(base_path+'/feature_map/{}'.format(j))
                            print(base_path+'/feature_map/{}'.format(j))
                        os.rename(img_path+img_name.group(),
                                  base_path+'/feature_map/'+img_number+'/'+img)


if __name__ == "__main__":
    mode = int(sys.argv[1])
    try:
        sorter(mode)
    except Exception as e:
        print(e)
