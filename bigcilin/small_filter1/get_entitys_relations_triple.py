# -*- coding:utf-8 -*-

import sys, os

in_file_name = "input.txt"
entity2id_file_name = "entity2id.txt"
relation2id_file_name = "relation2id.txt"
triple_file_name = "triple.txt"

if len(sys.argv) > 1:
    in_file_name = sys.argv[1]

entitys_set = set()
relations_set = set()
entity_relation_tuple = set()

fin = open(in_file_name, 'r')
fe = open(entity2id_file_name, 'w')
fr = open(relation2id_file_name, 'w')
ft = open(triple_file_name, 'w')
line = fin.readline()
while line:
    entity1, relation, entity2 = line.split('--->')
    entity1 = entity1.strip().replace(' ','').replace('　','')
    relation = relation.strip().replace(' ','').replace('　','')
    entity2 = entity2.strip().replace(' ','').replace('　','')
    
    if (entity1, relation) not in entity_relation_tuple:
        ft.write("%s\t%s\t%s\n" % (entity1, entity2, relation))
        entity_relation_tuple.add((entity1, relation))
        entitys_set.add(entity1)
        relations_set.add(relation)
        entitys_set.add(entity2)
        
    line = fin.readline()

index = 0
for entity in entitys_set:
    fe.write("%s\t%d\n" % (entity, index))
    index += 1
    
index = 0
for relation in relations_set:
    fr.write("%s\t%d\n" % (relation, index))
    index += 1

fin.close()
fe.close()
fr.close()
ft.close()
