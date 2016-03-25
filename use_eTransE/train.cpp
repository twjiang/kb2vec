/*************************************************
Author: Tianwen Jiang
Date: 2016-01-24
Description: the implementation of TransE (paper:Translating Embeddings for Modeling Multi-relational Data)
**************************************************/

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <map>
#include <vector>

using namespace std;

#define pi 3.1415926535897932384626433832795

map<string, int> entity2id, relation2id;
map<int, string> id2entity, id2relation;
int entity_num, relation_num;

/* the number of tail or head for a head or tail entity in the given relation */
map<int, map<int,int> > head_entity, tail_entity;
map<int, double> tph, hpt;

char buf[100000];
int L1_flag = 0;
string version = "unif";
string data_size = "all";

int rand_max(int x)
{
    int res = (rand()*rand())%x;
    while (res<0)
        res+=x;
    return res;
}
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double length=0;
    for (unsigned i=0; i<a.size(); i++)
		length+=a[i]*a[i];
	length = sqrt(length);
	return length;
}

void vec_norm(vector<double> &a)
{
    double x = vec_len(a);
    if (x>1)
        for (unsigned i=0; i<a.size(); i++)
            a[i]/=x;
}

class Train
{
private:
    vector<int> triple_h, triple_r, triple_t;
    /* record a triple whether in train triple */
    map<pair<int,int>, map<int,int> > in_train;

    int n, margin, method;
    double rate, loss_value;

    vector<vector<double> > entity_vec, relation_vec, entity_vec_tmp, relation_vec_tmp;

    void gradient(int h1,int r1,int t1,int h2,int r2,int t2)
    {
        for (int ii=0; ii<n; ii++)
        {

            double napla = 2*(entity_vec[h1][ii]+relation_vec[r1][ii]-entity_vec[t1][ii]);
            if (L1_flag)
            {
                if (napla>0)
            		napla=1;
            	else
            		napla=-1;
            }
            else
                napla = napla;
            relation_vec_tmp[r1][ii] -= rate*napla;
            entity_vec_tmp[h1][ii] -= rate*napla;
            entity_vec_tmp[t1][ii] -= -1*rate*napla;

            napla = 2*(entity_vec[h2][ii]+relation_vec[r2][ii]-entity_vec[t2][ii]);
            if (L1_flag)
            {
                if (napla>0)
            		napla=1;
            	else
            		napla=-1;
            }
            else
                napla = napla;
            relation_vec_tmp[r2][ii] -= -1*rate*napla;
            entity_vec_tmp[h2][ii] -= -1*rate*napla;
            entity_vec_tmp[t2][ii] -= rate*napla;
        }
    }

    double calc_distance(int h, int r, int t)
    {
        double distance = 0;
        if (L1_flag == 1)
            for (int i = 0; i < n; i++)
                distance += fabs(entity_vec[h][i] + relation_vec[r][i] - entity_vec[t][i]);
        else
            for (int i = 0; i < n; i++)
                distance += sqr(entity_vec[h][i] + relation_vec[r][i] - entity_vec[t][i]);
        return distance;
    }

    void train_kb(int h1, int r1, int t1, int h2, int r2, int t2)
    {
        double positive_distance = calc_distance(h1, r1, t1);
        double negtive_distance = calc_distance(h2, r2, t2);
        if (margin+positive_distance > negtive_distance)
        {
            loss_value += margin + positive_distance - negtive_distance;
            gradient(h1, r1, t1, h2, r2, t2);
        }
    }

    void train_model()
    {
        int nbatches = 100;
        int batch_size = triple_h.size() / nbatches;
        int nepochs = 1000;
        
        cout << "[info] step 5" << endl;
        
        for (int epoch = 0; epoch < nepochs; epoch++)
        {
            loss_value = 0;
            for (int batch = 0; batch < nbatches; batch++)
            {
                entity_vec_tmp = entity_vec;
                relation_vec_tmp = relation_vec;
                for (int k = 0; k < batch_size; k++)
                {   
                    int j = rand_max(triple_h.size());
                    int negtive_id = rand_max(entity_num);
                    int negtive_r_id = rand_max(relation_num);

                    double pr = 0;
                    if (method == 1)
                        pr = 1000 * tph[triple_r[j]]/(tph[triple_r[j]]+hpt[triple_r[j]]);
                    else
                        pr = 500;

                    if (rand()%1000 < pr)
                    {
                        while (in_train[make_pair(negtive_id,triple_r[j])].count(triple_t[j])>0)
                            negtive_id = rand_max(entity_num);

                        train_kb(triple_h[j], triple_r[j], triple_t[j], negtive_id, triple_r[j], triple_t[j]);
                    }
                    else
                    {
                        while (in_train[make_pair(triple_h[j],triple_r[j])].count(negtive_id)>0)
                            negtive_id = rand_max(entity_num);

                        train_kb(triple_h[j], triple_r[j], triple_t[j], triple_h[j], triple_r[j], negtive_id);
                    }
                    
                    while (in_train[make_pair(triple_h[j],negtive_r_id)].count(triple_t[j])>0)
                            negtive_r_id = rand_max(relation_num);

                    train_kb(triple_h[j], triple_r[j], triple_t[j], triple_h[j], negtive_r_id, triple_t[j]);

                    vec_norm(entity_vec_tmp[triple_h[j]]);
                    vec_norm(entity_vec_tmp[triple_t[j]]);
                    vec_norm(entity_vec_tmp[negtive_id]);
                    vec_norm(relation_vec_tmp[triple_r[j]]);
                }
                //cout << loss_value << endl;
                entity_vec = entity_vec_tmp;
                relation_vec = relation_vec_tmp;
                //cout << batch << endl;
            }
            cout << "[info] epoch:" << epoch <<'\t'<< loss_value <<endl;
        }
        FILE* relation_vec_file = fopen(("relation2vec."+version).c_str(),"w");
        FILE* entity_vec_file = fopen(("entity2vec."+version).c_str(),"w");
        cout << "[info] ... writing entity_vec and relation_vec to file." << endl;
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(relation_vec_file,"%.6lf\t",relation_vec[i][ii]);
            fprintf(relation_vec_file,"\n");
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(entity_vec_file,"%.6lf\t",entity_vec[i][ii]);
            fprintf(entity_vec_file,"\n");
        }
        cout << "[info] writed entity_vec and relation_vec to file." << endl;
        fclose(relation_vec_file);
        fclose(entity_vec_file);
    }

public:
    /* the start of the training: initialize the vector */
    void run(int n_in, int margin_in, int method_in, double rate_in)
    {
        n = n_in;
        margin = margin_in;
        method = method_in;
        rate = rate_in;
        
        //cout << "[info] step 1" << endl;
        
        entity_vec.resize(entity_num);
        for (unsigned i = 0; i < entity_vec.size(); i++)
            entity_vec[i].resize(n);
        relation_vec.resize(relation_num);
        for (unsigned i = 0; i < relation_vec.size(); i++)
            relation_vec[i].resize(n);
        
        //cout << "[info] step 2" << endl;
        
        entity_vec_tmp.resize(entity_num);
        for (unsigned i = 0; i < entity_vec_tmp.size(); i++)
            entity_vec_tmp[i].resize(n);
        relation_vec_tmp.resize(relation_num);
        for (unsigned i = 0; i < relation_vec_tmp.size(); i++)
            relation_vec_tmp[i].resize(n);
        
        cout << "[info] ... initilazing entity_vec and relation_vec." << endl;

        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            vec_norm(entity_vec[i]);
            if ((i % 10000) == 0)
                cout << "[info] entity_vec init " << i << " done" << endl;
        }
        
        cout << "[info] step 4" << endl;

        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            if ((i % 10000) == 0)
                cout << "[info] relation_vec init " << i << " done" << endl;
        }

        train_model();
    }

    /* add train triple for later training */
    void add (int head_id, int rel_id, int tail_id)
    {
        triple_h.push_back(head_id);
        triple_r.push_back(rel_id);
        triple_t.push_back(tail_id);
        in_train[make_pair(head_id, rel_id)][tail_id] = 1;
    }
};

Train train;

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

/*************************************************
Function: load_train_data()
Description: load train data from file
Input:
Output:
Return:
Others:
*************************************************/
void load_train_data()
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
            cout << "no relation: " << r << endl;

        head_entity[relation2id[r]][entity2id[h]]++;
        tail_entity[relation2id[r]][entity2id[t]]++;
        train.add(entity2id[h], relation2id[r], entity2id[t]);
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
int main(int argc,char **argv)
{
    int n, margin, method, ds;
    double rate;

    n = 100;
    margin = 1;
    method  = 1;
    rate = 0.01;

    int index;
    if ((index = have_arg((char *)"-size", argc, argv)) > 0) n = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-method", argc, argv)) > 0) method = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-rate", argc, argv)) > 0) rate = atof(argv[index+1]);
    if ((index = have_arg((char *)"-L1", argc, argv)) > 0) L1_flag = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-ds", argc, argv)) > 0) data_size = argv[index+1];

    cout << "size = " << n << endl;
    cout << "margin = " << margin << endl;
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
    cout << "rate = " << rate << endl;
    cout << "L1_flag = " << L1_flag << endl;
    cout << "data_size = " << data_size << endl;

    load_entity_relation_data();
    load_train_data();

    train.run(n, margin, method, rate);

    return 0;
}
