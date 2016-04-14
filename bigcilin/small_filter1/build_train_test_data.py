# -*- coding:utf-8 -*-

import sys, os
import random

in_file_name = "triple.txt"
train_file_name = "train.txt"
test_file_name = "test.txt"

entity_num_dict = dict()
relation_num_dict = dict()

if len(sys.argv) > 1:
    in_file_name = sys.argv[1]

fin = open(in_file_name, 'r')

line_num = 0
line = fin.readline()
while line:
    entity1, entity2, relation = line.strip().split('\t')
    try:
        entity_num_dict[entity1] += 1
    except:
        entity_num_dict[entity1] = 1
    try:
        entity_num_dict[entity2] += 1
    except:
        entity_num_dict[entity2] = 1
    try:
        relation_num_dict[relation] += 1
    except:
        relation_num_dict[relation] = 1
    line_num += 1
    line = fin.readline()
fin.close()

fin = open(in_file_name, 'r')
ftrain = open(train_file_name, 'w')
ftest = open(test_file_name, 'w')

hit_line_num = random.sample(xrange(line_num), 9000)

line_num = 0
line = fin.readline()
while line:
    entity1, entity2, relation = line.strip().split('\t')
    if line_num in hit_line_num:
        if entity_num_dict[entity1] > 1 and entity_num_dict[entity2] > 1 and relation_num_dict[relation] > 1:
            ftest.write("%s\t%s\t%s\n" % (entity1, entity2, relation))
            ftest.flush()
            entity_num_dict[entity1] -= 1
            entity_num_dict[entity2] -= 1
            relation_num_dict[relation] -= 1
        else:
            ftrain.write("%s\t%s\t%s\n" % (entity1, entity2, relation))
    else:
        ftrain.write("%s\t%s\t%s\n" % (entity1, entity2, relation))
    line = fin.readline()
    line_num += 1
fin.close()

ftrain.close()
ftest.close()
