#!/usr/bin/env python

import os
import shutil

delete = False
for bnn_name in os.listdir('.'):
    print(bnn_name)
    if os.path.isdir(bnn_name):
        for subdir in os.listdir(bnn_name):
            if subdir != 'train':
                to_delete = os.path.join(bnn_name, subdir)
                print('Deleting {}'.format(to_delete))
                if os.path.isdir(to_delete):
                    shutil.rmtree(to_delete)
                    delete = True
                elif os.path.isfile(to_delete):
                    print('Deleting {}'.format(to_delete))
                    os.remove(to_delete)

if not delete:
    print('Already clean - contains only trained models.')
