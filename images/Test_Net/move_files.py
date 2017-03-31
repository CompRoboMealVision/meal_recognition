import os

i = 0
for filename in os.listdir('.'):
    if filename.endswith('jpg'):
        new_name = 'Apple_Pie/' + str(i) + '.jpg'
        os.rename(filename, new_name)
        print filename, new_name
        i = i + 1