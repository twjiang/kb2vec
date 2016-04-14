/*************************************************
Author: Tianwen Jiang
Date: 2016-01-26
Description: the Test for kb_embedding program
**************************************************/

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace std;

typedef pair<int, double> PAIR;  
  
bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {  
  return lhs.second < rhs.second;  
}  
  
struct CmpByValue {  
  bool operator()(const PAIR& lhs, const PAIR& rhs) {  
    return lhs.second < rhs.second;  
  }  
}; 

vector<int> triple_h, triple_r, triple_t;
map<string, int> entity2id, relation2id;
map<int, string> id2entity, id2relation;
map<string, int> entity2count, relation2count;
int entity_num, relation_num;

/* the number of tail or head for a head or tail entity in the given relation */
map<int, map<int,int> > head_entity, tail_entity;
map<int, double> tph, hpt;

vector<vector<double> > entity_vec, relation_vec;
map<pair<int, int>, map<int, int> > is_good_triple;

char buf[100000];
int L1_flag = 0;
int n = 100;
string version;
string data_size = "all";
string model = "model_small_1";

double sqr(double x)
{
    return x*x;
}

/*************************************************
Function: load_test_data()
Description: load test data from file
Input:
Output:
Return:
Others:
*************************************************/
void load_test_data()
{
    ifstream test_file;
    test_file.open(("../bigcilin/"+data_size+"/test.txt").c_str());
    string h , r, t;
	while(!test_file.eof())
    {
        test_file >> h >> t >> r;

        if (entity2id.count(h)==0)
            cout << "no entity: " << h << endl;
        if (entity2id.count(t)==0)
            cout << "no entity: " << t << endl;

        if (relation2id.count(r)==0)
            cout << "no relation: " << t << endl;

        triple_h.push_back(entity2id[h]);
        triple_r.push_back(relation2id[r]);
        triple_t.push_back(entity2id[t]);
    }

    cout << "test triple num = " << triple_h.size() << endl;
    test_file.close();
}

void load_KB_data()
{
    ifstream train_file;
    train_file.open(("../bigcilin/"+data_size+"/train.txt").c_str());
    string h , r, t;
	while(!train_file.eof())
    {
        train_file >> h >> t >> r;

        if (entity2id.count(h)==0)
            cout << "no entity: " << h << endl;
        if (entity2id.count(t)==0)
            cout << "no entity: " << t << endl;

        if (relation2id.count(r)==0)
            cout << "no relation: " << t << endl;
        
        entity2count[h]++;
        entity2count[t]++;
        relation2count[r]++;
        
        head_entity[relation2id[r]][entity2id[h]]++;
        tail_entity[relation2id[r]][entity2id[t]]++;
        
        is_good_triple[make_pair(entity2id[h], relation2id[r])][entity2id[t]] = 1;
    }
    
    train_file.close();
    
    for (int i = 0; i < relation_num; i++)
    {
        double sort_num=0, total_num=0;
        for (map<int,int>::iterator it = head_entity[i].begin(); it != head_entity[i].end(); it++)
        {
            sort_num++;
            total_num += it->second;
        }
        tph[i] = total_num / sort_num;
    }

    for (int i = 0; i < relation_num; i++)
    {
        double sort_num=0, total_num=0;
        for (map<int,int>::iterator it = tail_entity[i].begin(); it != tail_entity[i].end(); it++)
        {
            sort_num++;
            total_num += it->second;
        }
        hpt[i] = total_num / sort_num;
    }
}

/*************************************************
Function: load_entity_relation_data()
Description: load entity and relation data from file
Input:
Output: the number of entity and relation
Return:
Others:
*************************************************/
void load_entity_relation_data()
{
    FILE *entity_file = fopen(("../bigcilin/"+data_size+"/entity2id.txt").c_str(),"r");
	FILE *relation_file = fopen(("../bigcilin/"+data_size+"/relation2id.txt").c_str(),"r");

    int id;
    const char * split = "\t";
    char * p;
    
	while(!feof(entity_file))
    {
        fgets(buf, sizeof(buf), entity_file); 
        
        p = strtok(buf, split);
        string s = p;
        //cout << s << endl;
        p = strtok(NULL, split);
        if (p==NULL)
            continue;
        id = atoi(p);
        //cout << id << endl;

        entity2id[s] = id;
        id2entity[id] = s;
        entity_num++;
    }
    cout << "number of entity = " << entity_num << endl;
    fclose(entity_file);

	while(!feof(relation_file))
    {
        fgets(buf, sizeof(buf), relation_file); 
        
        p = strtok(buf, split);
        string s = p;
        //cout << s << endl;
        p = strtok(NULL, split);
        if (p==NULL)
            continue;
        id = atoi(p);
        //cout << id << endl;
        relation2id[s] = id;
        id2relation[id] = s;
        relation_num++;
    }
    cout << "number of relation = " << relation_num << endl;
    fclose(relation_file);
}

void load_entity_relation_vec()
{
    FILE *relation_vec_file = fopen(("./model/"+model+"/relation2vec."+version).c_str(), "r");
    FILE *entity_vec_file = fopen(("./model/"+model+"/entity2vec."+version).c_str(), "r");

    relation_vec.resize(relation_num);
    for (int i=0; i<relation_num;i++)
    {
        relation_vec[i].resize(n);
        for (int ii=0; ii<n; ii++){
            fscanf(relation_vec_file, "%lf", &relation_vec[i][ii]);
            //cout << relation_vec[i][ii] << endl;
        }
        if (i%10000==0)
            cout << "[info] relation_vec load " << i << " done" << endl;
    }
    entity_vec.resize(entity_num);
    for (int i=0; i<entity_num;i++)
    {
        entity_vec[i].resize(n);
        for (int ii=0; ii<n; ii++)
            fscanf(entity_vec_file, "%lf", &entity_vec[i][ii]);
        if ((i % 100000) == 0)
            cout << "[info] entity_vec load " << i << " done" << endl;
    }
    fclose(relation_vec_file);
    fclose(entity_vec_file);
}

double calc_distance(int h, int r, int t)
{
    double distance = 0;
    if (L1_flag == 1)
        for (int i = 0; i < n; i++)
            distance += fabs(entity_vec[h][i] + relation_vec[r][i] - entity_vec[t][i]);
    else
        for (int i = 0; i < n; i++){
            distance += sqr(entity_vec[h][i] + relation_vec[r][i] - entity_vec[t][i]);
        }
    return distance;
}

/*************************************************
Function: link_prediction()
Description: predict the ranked_link_list for given entitys
Input:
    entity1_id: int
    entity2_id: int
Output:
Return: the ranked_link_map
Others:
*************************************************/
map<int, double> link_prediction(int entity1_id, int entity2_id, double threshold, int r_rank)
{
    map<int, double> ranked_link_map;
    double min = 10000000;
    int r_id = -1;
    double distance;
    
    for(int j = 0; j < r_rank; j++)
    {
        for(int i = 0; i < relation_num; i++)
        {
            if (is_good_triple[make_pair(entity1_id, i)].count(entity2_id) > 0){
                    continue;
            }
            if (ranked_link_map.find(i) != ranked_link_map.end())
                continue;
            distance = calc_distance(entity1_id, i, entity2_id);
            if(distance < min)
            {
                min = distance;
                r_id = i;
            }
        }
        if (min >= threshold)
            break;
        ranked_link_map[r_id] = min;
        min = 10000000;        
    }
    
    return ranked_link_map;
}

/*************************************************
Function: entity_prediction()
Description: predict the ranked_link_list for given entitys
Input:
    entity1_id: int
    r_id: int
Output:
Return: the ranked_entity_map
Others:
*************************************************/
map<int, double> entity_prediction(int entity1_id, int r_id, double threshold, int e_rank, int r_rank)
{
    map<int, double> predict_entity_map_tmp;
    map<int, double> predict_entity_map;
    map<int, double> ranked_map;
    vector<PAIR> result_score_vec;
    int min_id = -1;
    double min = 10000000;
    double distance;
    
    for(int j = 0; j < entity_num; j++)
    {
        if (is_good_triple[make_pair(entity1_id, r_id)].count(j) > 0){
                continue;
        }
        distance = calc_distance(entity1_id, r_id, j);
        if(distance < threshold)
        {
            ranked_map = link_prediction(entity1_id, j, threshold, r_rank);
            if (ranked_map.size()!=0)
            {
                if (ranked_map.find(r_id) != ranked_map.end())
                {
                    int rank = 1;
                    for (map<int, double>::iterator it=ranked_map.begin(); it!=ranked_map.end(); ++it)
                    {
                        if (it->first == r_id)
                            continue;
                        //cout << it->second << " " << ranked_map[r_id] << endl;
                        if (it->second < ranked_map[r_id])
                            rank ++;
                    }
                    predict_entity_map_tmp[j] = rank*distance;
                }
            }
            //predict_entity_map_tmp[j] = distance;
        }   
    }
    
    if (predict_entity_map_tmp.size() == 0)
        return predict_entity_map;
    
    for(int j = 0; j < e_rank; j++)
    {
        if (j == predict_entity_map_tmp.size())
            return predict_entity_map;
        for (map<int, double>::iterator it=predict_entity_map_tmp.begin(); it!=predict_entity_map_tmp.end(); ++it) {  
            if (predict_entity_map.find(it->first) != predict_entity_map.end())
                continue;
            if(it->second < min)
            {
                min = it->second;
                min_id = it->first;
            }
        }  
        predict_entity_map[min_id] = min;
        min = 10000000;
    } 
    
    return predict_entity_map;
}

double calc_entity_distance(int e1, int e2)
{
    double distance = 0;
    if (L1_flag == 1)
        for (int i = 0; i < n; i++)
            distance += fabs(entity_vec[e1][i] - entity_vec[e2][i]);
    else
        for (int i = 0; i < n; i++){
            distance += sqr(entity_vec[e1][i] - entity_vec[e2][i]);
        }
    return distance;
}

/*************************************************
Function: test_predict_entity()
Description: 
Input:
Output:
Return: 
Others:
*************************************************/
void test_predict_entity()
{
    map<int, double> ranked_map;
    vector<PAIR> result_score_vec;
    ofstream result_file;
    ofstream result_right_file;
    ofstream result_wrong_file1;
    ofstream result_wrong_file2;
    result_file.open(("./model/"+model+"/test_predict_entity.result").c_str());
    result_right_file.open(("./model/"+model+"/test_predict_entity_right.result").c_str());
    result_wrong_file1.open(("./model/"+model+"/test_predict_entity_wrong_1.result").c_str());
    result_wrong_file2.open(("./model/"+model+"/test_predict_entity_wrong_2.result").c_str());
    
    int hit_10_count = 0;
    int hit_1_count = 0;
    int count = 0;
    int simi_count = 0;
    
    for (int i = 0; i < triple_h.size(); i++)
    {
        result_score_vec.clear();
        ranked_map = entity_prediction(triple_h[i], triple_r[i], 1.0, 10, 10);
        if (ranked_map.size()!=0)
        {
            count++;
            
            int rank = 0;
            int hit_10 = 0;
            int hit_1 = 0;
            double simi = 0;
            
            result_file << id2entity[triple_h[i]] << "==" << id2entity[triple_t[i]] << "==" << id2relation[triple_r[i]];  
            result_file << ":" << calc_distance(triple_h[i], triple_r[i], triple_t[i]);
            for (map<int, double>::iterator it=ranked_map.begin(); it!=ranked_map.end(); ++it) {  
                result_score_vec.push_back(make_pair(it->first, it->second));  
            }  
            sort(result_score_vec.begin(), result_score_vec.end(), CmpByValue());     
            for (vector<PAIR>::iterator it=result_score_vec.begin(); it!=result_score_vec.end(); ++it) {  
                result_file << "##" << id2entity[it->first] << ":" << it->second;  
                rank++;
                if (rank == 1)
                {
                    simi = calc_entity_distance(it->first, triple_t[i]);
                    if (simi < 0.985)
                        simi_count++;
                }
                if (it->first == triple_t[i])
                {
                    hit_10 = 1;
                    if (rank == 1)
                        hit_1 = 1;
                }
            }
            if (hit_10)
            {
                hit_10_count++;
                result_right_file << id2entity[triple_h[i]] << "==" << id2entity[triple_t[i]] << "==" << id2relation[triple_r[i]];
                result_right_file << "##" << entity2count[id2entity[triple_h[i]]] << "##" << entity2count[id2entity[triple_t[i]]] << "##" << relation2count[id2relation[triple_r[i]]] << "##" << tph[triple_r[i]]/(tph[triple_r[i]]+hpt[triple_r[i]]) << endl;
            }
            else
            {
                result_wrong_file1 << id2entity[triple_h[i]] << "==" << id2entity[triple_t[i]] << "==" << id2relation[triple_r[i]];
                result_wrong_file1 << "##" << entity2count[id2entity[triple_h[i]]] << "##" << entity2count[id2entity[triple_t[i]]] << "##" << relation2count[id2relation[triple_r[i]]] << "##" << tph[triple_r[i]]/(tph[triple_r[i]]+hpt[triple_r[i]]) << endl;
            }
            
            if (hit_1)
                hit_1_count++;
            
            result_file << "##" << hit_10 << "##" << hit_1 << "##hit_10_p:" << ((double)hit_10_count / count)*100 << "% hit_1_p:" << ((double)hit_1_count / count)*100 << "%" << " hit_simi_p["<< simi <<"]:" << ((double)simi_count / count)*100 << "%"; 
            
            result_file << endl;
        }
        else
        {
            result_wrong_file2 << id2entity[triple_h[i]] << "==" << id2entity[triple_t[i]] << "==" << id2relation[triple_r[i]];
            result_wrong_file2 << "##" << entity2count[id2entity[triple_h[i]]] << "##" << entity2count[id2entity[triple_t[i]]] << "##" << relation2count[id2relation[triple_r[i]]] << "##" << tph[triple_r[i]]/(tph[triple_r[i]]+hpt[triple_r[i]]) << endl;
        }
        if (i % 10 == 0)
        {
            cout << i << "/" << triple_h.size() <<" done." << endl;
        }
    }
    
    result_file.close();
    result_wrong_file1.close();
    result_wrong_file2.close();
    result_right_file.close();
}

/*************************************************
Function: have_arg()
Description: acquire the value for the given arg name
Input:
    char *str: the given arg name
    int argc: the number of the arguments
    char **argv: the arguments
Output:
Return:
    int index: the index of the value for the given arg name
Others:
*************************************************/
int have_arg(char *str, int argc, char**argv)
{
    for (int index = 0; index < argc; index++)
    {
        if (!strcmp(str, argv[index]))
        {
            if (index == argc - 1)
            {
                cout << "no such argument!" << endl;
                exit(1);
            }
            return index;
        }
    }
    return -1;
}

/*************************************************
Function: main()
Description: main control program
Input:
    int argc: the number of the arguments
    char **argv: the arguments
Output: the setting of model
Return:
    the state of excution
Others:
*************************************************/
int main(int argc, char **argv)
{
    int index;
    int method = 1;
    if ((index = have_arg((char *)"-size", argc, argv)) > 0) n = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-method", argc, argv)) > 0) method = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-L1", argc, argv)) > 0) L1_flag = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-ds", argc, argv)) > 0) data_size = argv[index+1];
    if ((index = have_arg((char *)"-model", argc, argv)) > 0) model = argv[index+1];

    cout << "size = " << n << endl;
    if (method == 1)
    {
        cout << "method = " << "bern" << endl;
        version = "bern";
    }
    else
    {
        cout << "method = " << "unif" << endl;
        version = "unif";
    }
    if (L1_flag == 1)
        cout << "use L1 to calculate distance." << endl;
    else
        cout << "use L2 to calculate distance." << endl;

    load_entity_relation_data();
    
    cout << "load entity and relation embedding data ... ..." << endl;
    load_entity_relation_vec();
    cout << "load ok." << endl;
    load_KB_data();
    cout << "load test data ... ..." << endl;
    load_test_data();
    cout << "load ok." << endl;
    
    cout << "testing for predicting entity ... ..." << endl;
    
    test_predict_entity();

    return 0;
}
