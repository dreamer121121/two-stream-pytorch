import  os
path = '/home/xy/data/jpegs_256/'
videos = os.listdir(path)
classdir = []
for v in videos:
    name = v.split('_')[1]
    if name not in classdir:
        os.mkdir(path+name)
        classdir.append(name)
    os.system('mv '+v+' '+name)
print(classdir)

