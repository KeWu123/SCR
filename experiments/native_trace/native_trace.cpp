#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <omp.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
template<class T> void arr(std::ostream& o,const std::vector<T>& a){o<<'[';for(size_t i=0;i<a.size();i++){if(i)o<<',';o<<a[i];}o<<']';}
template<class T> void read(std::istream& f,std::vector<T>& a){f.read((char*)a.data(),a.size()*sizeof(T));require(bool(f),"binary input truncated");}
int shared(int v,const std::vector<int>& map){return v<0?v:map.at(v);}

void save_graph(const std::filesystem::path& path,const char* name,const faiss::IndexHNSWFlat& idx,const std::vector<int>& map){
    std::ofstream out(path);require(bool(out),"graph output failed");
    auto& h=idx.hnsw;
    out<<"{\"generation\":\""<<name<<"\",\"entry_point\":"<<shared(h.entry_point,map)<<",\"max_level\":"<<h.max_level<<",\"nodes\":[";
    for(int local=0;local<idx.ntotal;local++){
        if(local)out<<',';
        out<<"{\"id\":"<<map[local]<<",\"layers\":[";
        for(int level=0;level<h.levels[local];level++){
            if(level)out<<',';
            size_t b,e;h.neighbor_range(local,level,&b,&e);
            std::vector<int> slots;for(size_t j=b;j<e;j++)slots.push_back(shared(h.neighbors[j],map));
            out<<"{\"level\":"<<level<<",\"slots\":";arr(out,slots);out<<'}';
        }
        out<<"]}";
    }
    out<<"]}";
    require(bool(out),"graph write failed");
}

void save_search(std::ostream& out,int seed,int ef,int qi,const char* name,
                 faiss::IndexHNSWFlat& idx,const float* query,const std::vector<int>& map){
    constexpr int K=10;
    std::vector<faiss::idx_t> labels_off(K),labels_on(K);
    std::vector<float> distance_off(K),distance_on(K);
    idx.search(1,query,K,distance_off.data(),labels_off.data());
    std::vector<faiss::SCRNativeExpansion> expansions;
    std::vector<faiss::SCRNativeVisit> visits;
    faiss::scr_native_expansions=&expansions;faiss::scr_native_visits=&visits;
    try{idx.search(1,query,K,distance_on.data(),labels_on.data());}
    catch(...){faiss::scr_native_expansions=nullptr;faiss::scr_native_visits=nullptr;throw;}
    faiss::scr_native_expansions=nullptr;faiss::scr_native_visits=nullptr;
    require(std::memcmp(labels_off.data(),labels_on.data(),K*sizeof(faiss::idx_t))==0,"trace changed labels");
    require(std::memcmp(distance_off.data(),distance_on.data(),K*sizeof(float))==0,"trace changed distances");
    require(!expansions.empty(),"native traversal trace empty");
    require(!visits.empty(),"native first-visited trace empty");
    std::vector<int> ids_off,ids_on;std::vector<uint32_t> bits_off,bits_on;
    for(int j=0;j<K;j++){
        require(labels_on[j]>=0&&labels_on[j]<idx.ntotal,"native label invalid");
        ids_off.push_back(shared(int(labels_off[j]),map));ids_on.push_back(shared(int(labels_on[j]),map));
        bits_off.push_back(std::bit_cast<uint32_t>(distance_off[j]));bits_on.push_back(std::bit_cast<uint32_t>(distance_on[j]));
    }
    out<<"{\"seed\":"<<seed<<",\"ef\":"<<ef<<",\"query\":"<<qi<<",\"generation\":\""<<name<<"\",\"labels_off\":";arr(out,ids_off);
    out<<",\"labels_on\":";arr(out,ids_on);out<<",\"distance_bits_off\":";arr(out,bits_off);out<<",\"distance_bits_on\":";arr(out,bits_on);
    out<<",\"expansions\":[";
    for(size_t j=0;j<expansions.size();j++){
        if(j)out<<',';auto& x=expansions[j];std::vector<int> slots;
        for(int v:x.slots)slots.push_back(shared(v,map));
        out<<'['<<x.level<<','<<shared(x.source,map)<<',';arr(out,slots);out<<']';
    }
    out<<"],\"first_visited_edges\":[";
    for(size_t j=0;j<visits.size();j++){
        if(j)out<<',';auto& v=visits[j];out<<'['<<v.level<<','<<shared(v.source,map)<<','<<shared(v.target,map)<<']';
    }
    out<<"]}\n";require(bool(out),"raw trace write failed");
}

int main(int argc,char** argv){try{
    require(argc==2,"pass output directory");std::filesystem::path root=std::filesystem::u8path(argv[1]);
    omp_set_num_threads(1);
    for(int seed:{11,29,47}){
        std::ifstream f(root/("input_"+std::to_string(seed)+".bin"),std::ios::binary);require(bool(f),"input open failed");
        std::vector<int> header(4);read(f,header);int N=header[0],D=header[1],Q=header[2],R=header[3];
        require(N==2000&&D==32&&Q==200&&R==100,"protocol dimensions changed");
        std::vector<float> xb(N*D),queries(Q*D),xb_permuted(N*D);std::vector<int> perm(N),rewritten(R),identity(N),inv(N);
        read(f,xb);read(f,queries);read(f,perm);read(f,rewritten);std::iota(identity.begin(),identity.end(),0);
        auto sorted=perm;std::sort(sorted.begin(),sorted.end());require(sorted==identity,"invalid permutation");
        for(int local=0;local<N;local++){
            inv[perm[local]]=local;
            std::copy(xb.begin()+perm[local]*D,xb.begin()+(perm[local]+1)*D,xb_permuted.begin()+local*D);
        }
        faiss::IndexHNSWFlat A(D,16),B(D,16),C(D,16);
        for(auto* idx:{&A,&B,&C})idx->hnsw.efConstruction=64;
        A.add(N,xb.data());B.add(N,xb_permuted.data());C.add(N,xb.data());
        require(A.hnsw.neighbors==C.hnsw.neighbors&&A.hnsw.levels==C.hnsw.levels&&A.hnsw.offsets==C.hnsw.offsets&&A.hnsw.entry_point==C.hnsw.entry_point,"A/C pre-rewrite not identical");
        std::vector<float> restored(N*D);A.reconstruct_n(0,N,restored.data());require(std::memcmp(xb.data(),restored.data(),xb.size()*sizeof(float))==0,"A vectors changed");
        B.reconstruct_n(0,N,restored.data());require(std::memcmp(xb_permuted.data(),restored.data(),restored.size()*sizeof(float))==0,"B vectors changed");
        for(int u:rewritten){size_t ab,ae,bb,be;C.hnsw.neighbor_range(u,0,&ab,&ae);B.hnsw.neighbor_range(inv[u],0,&bb,&be);require(ae-ab==be-bb,"base capacities differ");
            for(size_t j=0;j<ae-ab;j++)C.hnsw.neighbors[ab+j]=shared(B.hnsw.neighbors[bb+j],perm);
        }
        save_graph(root/("graph_"+std::to_string(seed)+"_A.json"),"A",A,identity);
        save_graph(root/("graph_"+std::to_string(seed)+"_B.json"),"B",B,perm);
        save_graph(root/("graph_"+std::to_string(seed)+"_C.json"),"C",C,identity);
        // ---- maintenance generation: replace the last 25% of vectors, then rebuild ----
        std::vector<float> xb_maint(N*D);std::copy(xb.begin(),xb.end(),xb_maint.begin());
        std::mt19937 rng_maint(seed+999);std::uniform_real_distribution<float> ud(0.0f,1.0f);
        for(int i=1500*D;i<N*D;i++) xb_maint[i]=ud(rng_maint);
        faiss::IndexHNSWFlat Maint(D,16);Maint.hnsw.efConstruction=64;Maint.add(N,xb_maint.data());
        std::vector<float> restoredD(N*D);Maint.reconstruct_n(0,N,restoredD.data());
        require(std::memcmp(xb_maint.data(),restoredD.data(),restoredD.size()*sizeof(float))==0,"Maint vectors changed");
        save_graph(root/("graph_"+std::to_string(seed)+"_D.json"),"D",Maint,identity);
        std::ofstream out(root/("raw_"+std::to_string(seed)+".jsonl"));require(bool(out),"raw open failed");
        for(int ef:{16,64,128}){
            A.hnsw.efSearch=ef;B.hnsw.efSearch=ef;C.hnsw.efSearch=ef;Maint.hnsw.efSearch=ef;
            for(int qi=0;qi<Q;qi++){
                auto* q=queries.data()+qi*D;
                save_search(out,seed,ef,qi,"A",A,q,identity);save_search(out,seed,ef,qi,"B",B,q,perm);
                save_search(out,seed,ef,qi,"C",C,q,identity);save_search(out,seed,ef,qi,"AA",A,q,identity);
                save_search(out,seed,ef,qi,"D",Maint,q,identity);
            }
            out.flush();std::cout<<"seed "<<seed<<" ef "<<ef<<" completed "<<Q<<" native traced queries per generation"<<std::endl;
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 2;}}
