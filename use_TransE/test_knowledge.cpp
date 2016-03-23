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

using namespace std;

map<string, int> entity2id, relation2id;
map<int, string> id2entity, id2relation;
int entity_num, relation_num;

map<pair<int, int>, map<int, int> > is_good_triple;

vector<int> triple_h, triple_r, triple_t;
vector<vector<double> > entity_vec, relation_vec;

char buf[100000];
int L1_flag = 0;
int n = 100;
string version;
string data_size = "all";

double sqr(double x)
{
    return x*x;
}

/*************************************************
Function: load_kb_data()
Description: load train data from file
Input:
Output:
Return:
Others:
*************************************************/
void load_kb_data()
{
    FILE *test_file = fopen(("../bigcilin/"+data_size+"/train.txt").c_str(),"r");
    
    const char * split = "\t";
    char * p;
	while(!feof(test_file))
    {
        fgets(buf, sizeof(buf), test_file); 
        
        p = strtok(buf, split);
        string h = p;
        p = strtok(NULL, split);
        if (p==NULL)
            continue;
        string t = p;
        p = strtok(NULL, split);
        string r = p;
        r = r.substr(0, r.length()-1);// minus 1 for Windows, minus 2 for Unix

        if (entity2id.count(h)==0)
            cout << "no entity: " << h << endl;
        if (entity2id.count(t)==0)
            cout << "no entity: " << t << endl;

        if (relation2id.count(r)==0)
            cout << "no relation: " << t << endl;

        triple_h.push_back(entity2id[h]);
        triple_r.push_back(relation2id[r]);
        triple_t.push_back(entity2id[t]);

        is_good_triple[make_pair(entity2id[h], relation2id[r])][entity2id[t]] = 1;
    }

    cout << "test triple num = " << triple_h.size() << endl;
    fclose(test_file);
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
    FILE *entity_file = fopen("../data/entity2id.txt","r");
	FILE *relation_file = fopen("../data/relation2id.txt","r");

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
    FILE *relation_vec_file = fopen(("./model/model_e2/relation2vec."+version).c_str(), "r");
    FILE *entity_vec_file = fopen(("./model/model_e2/entity2vec."+version).c_str(), "r");

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
Description: test for link prediction
Input:

Output:
Return:
Others:
*************************************************/
void link_prediction()
{
    int h_id, r_id, t_id, before_count, neg_count, rank1;
    double pos_distance, neg_distance;
    int mean_rank;
    double hit_10;
    int rank_count = 0;
    int hit_10_count = 0;
    int rank_sum = 0;
    
    ofstream bad_rank_triple_file;
    bad_rank_triple_file.open("bad_rank_triple.txt",ios::app);

    cout << "rank1\t" << "mean_rank\t" << "hit_10" << endl;
    for (int i = 0; i < triple_h.size(); i++)
    {
        h_id = triple_h[i];
        r_id = triple_r[i];
        t_id = triple_t[i];
        pos_distance = calc_distance(h_id, r_id, t_id);

        before_count = 0;
        neg_count = 0;
        for (int j = 0; j < entity_num; j++)
        {
            if (is_good_triple[make_pair(j, r_id)].count(t_id) > 0){
                continue;
            }
            else
            {
                neg_distance = calc_distance(j, r_id, t_id);
                //cout << pos_distance << "\t" << neg_distance << endl;
                neg_count++;
                if (neg_distance < pos_distance)
                    before_count++;
            }
        }
        //cout << "--" << endl;
        rank_count++;
        rank1 = before_count + 1;
        rank_sum += rank1;

        if (rank1 <= 10)
            hit_10_count++;

        mean_rank = rank_sum / rank_count;
        hit_10 = ((double)hit_10_count / rank_count)*100;
        cout << rank1 << "\t" << mean_rank << "\t" << hit_10 << "%" << endl;
        if (rank1 > 10){
            bad_rank_triple_file << id2entity[h_id] << "\t" << id2entity[t_id] << "\t" <<id2relation[r_id] << "\t" << rank1 << endl;
            bad_rank_triple_file.flush();
        }

        before_count = 0;
        neg_count = 0;
        for (int j = 0; j < entity_num; j++)
        {
            if (is_good_triple[make_pair(h_id, r_id)].count(j) > 0)
                continue;
            else
            {
                neg_distance = calc_distance(h_id, r_id, j);
                //cout << pos_distance << "\t" << neg_distance << endl;
                neg_count++;
                if (neg_distance < pos_distance)
                    before_count++;
            }
        }
        rank_count++;
        rank1 = before_count + 1;
        rank_sum += rank1;

        if (rank1 <= 10)
            hit_10_count++;

        mean_rank = rank_sum / rank_count;
        hit_10 = ((double)hit_10_count / rank_count)*100;
        cout << rank1 << "\t" << mean_rank << "\t" << hit_10 << "%" << endl;
        if (rank1 > 10){
            bad_rank_triple_file << id2entity[h_id] << "\t" << id2entity[t_id] << "\t" <<id2relation[r_id] << "\t" << rank1 << endl;
            bad_rank_triple_file.flush();
        }
    }
    bad_rank_triple_file.close();
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
    ds = 0;
    
    if ((index = have_arg((char *)"-size", argc, argv)) > 0) n = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-method", argc, argv)) > 0) method = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-L1", argc, argv)) > 0) L1_flag = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-ds", argc, argv)) > 0) ds = atoi(argv[index+1]);

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
    
    if (ds == 0)
    {
        cout << "data_size = " << "small" << endl;
        data_size = "small";
    }
    else
    {
        cout << "data_size = " << "all" << endl;
        data_size = "all";
    }

    load_entity_relation_data();
    load_kb_data();

    cout << "load entity and relation embedding data ... ..." << endl;
    load_entity_relation_vec();
    cout << "load ok." << endl;

    link_prediction();

    return 0;
}
