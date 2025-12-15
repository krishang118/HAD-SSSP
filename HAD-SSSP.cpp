#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <numeric>
#include <cassert>
#include <mutex>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace chrono;

static const int INF = numeric_limits<int>::max();

struct Edge { int to; int w; };

struct Node {
    int id{-1};
    double x{0}, y{0};
    vector<Edge> edges;
    int degree{0};
    int ownerRank{-1};};

struct L1Cluster {
    int id{-1};
    vector<int> nodes;
    double centerX{0}, centerY{0};
    int representative{-1};
    int ownerRank{-1};
    bool isBoundary{false};};

struct L2Cluster {
    int id{-1};
    vector<int> l1Clusters;
    double centerX{0}, centerY{0};
    int representative{-1};};

struct SuperNode {
    int id{-1};
    vector<int> l2Clusters;
    double centerX{0}, centerY{0};
    int representative{-1};};

struct DistanceResult {
    int distance{INF};
    bool isApproximate{true};
    double computationTime{0};
    int hierarchyLevel{3};};

struct VerificationStats {
    int totalQueries{0};
    int matchedQueries{0};
    int mismatchedQueries{0};
    double maxError{0.0};
    double avgError{0.0};
    vector<tuple<int,int,int,int>> mismatches;};

static inline int owner_of(int nodeId, int nodesPerRank, int worldSize) {
    if (nodeId < 0) return -1;
    int owner = nodeId / nodesPerRank;
    if (owner < 0) owner = 0;
    if (owner >= worldSize) owner = worldSize - 1;
    return owner;}

static inline void allreduce_or(int localFlag, int &globalFlag) {
    MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);}

class HADSSSP {
    int rank, size;
    int maxNodeId{-1};
    int nodesPerRank{1};
    bool isScaleFree{false}, isRoadNetwork{false};
    double avgDegree{0};

    unordered_map<int, Node> localGraph;
    unordered_set<int> ghostSet;

    vector<L1Cluster> allL1;
    vector<L2Cluster> allL2;
    vector<SuperNode> allSN;

    vector<int> nodeToL1;
    vector<int> l1ToL2;
    vector<int> l2ToSN;

    int snPrecomputeK{32};
    vector<int> snDist;
    string filename;
    VerificationStats verifyStats;

public:
    HADSSSP(int r, int s) : rank(r), size(s) {}
    const unordered_map<int, Node>& getLocalGraph() const { return localGraph; }
    void loadAndDistributeGraph(const string& fname) {
        filename = fname;
        vector<int> flatPairs;
        vector<int> flatWeights;
        int numEdges = 0;

        if (rank == 0) {
            cout << "\n";
            cout << "Step 1: Loading and Distributing Graph\n";
            cout << "[DEBUG] Rank 0: Attempting to open file: " << filename << "\n";
            cout.flush();

            ifstream f(filename);
            if (!f.is_open()) {
                cerr << "Error: Cannot open " << filename << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            cout << "[DEBUG] Rank 0: File opened successfully\n";
            cout.flush();
            
            srand(42);
            const int MIN_WEIGHT = 10;   
            const int MAX_WEIGHT = 1000; 
            
            cout << "[INFO] Random weight range: " << (MIN_WEIGHT/100.0) << " to " << (MAX_WEIGHT/100.0) << "\n";
            cout << "[INFO] Detecting file format...\n";
            cout.flush();
            
            bool isDirected = false;
            unordered_map<long long, bool> edgeCheck;
            int checkCount = 0;
            const int CHECK_LIMIT = 1000;
            
            string line;
            vector<string> allLines;
            
            while (getline(f, line)) {
                if (!line.empty() && line[0] != '#') {
                    allLines.push_back(line);                    
                    if (checkCount < CHECK_LIMIT) {
                        istringstream iss(line);
                        string s; iss >> s;
                        if (s == "source" || s == "Source" || s == "FromNodeId" || s == "ToNodeId") continue;
                        try {
                            int u = stoi(s);
                            int v;
                            if (iss >> v) {
                                long long edgeKey = ((long long)min(u,v) << 32) | (long long)max(u,v);
                                if (edgeCheck.count(edgeKey)) {
                                    isDirected = false;
                                } else {
                                    edgeCheck[edgeKey] = true;
                                }
                                checkCount++;
                            }
                        } catch (...) {
                            continue;}}}}
            
            if (checkCount > 50 && edgeCheck.size() == checkCount) {
                isDirected = true;
            }
            
            cout << "[INFO] Detected format: " << (isDirected ? "DIRECTED" : "UNDIRECTED") << " graph\n";
            cout << "[INFO] Processing " << allLines.size() << " edge entries...\n";
            cout.flush();
            
            for (const auto& line : allLines) {
                if (line.empty()) continue;
                
                istringstream iss(line);
                string s; iss >> s;                
                if (s == "source" || s == "Source" || s == "FromNodeId" || s == "ToNodeId") continue;
                int u, v;
                double w = 1.0;
                
                try {
                    u = stoi(s);
                    if (!(iss >> v)) continue;
                    
                    if (iss >> w) {
                        w = w;
                    } else {
                        w = (MIN_WEIGHT + rand() % (MAX_WEIGHT - MIN_WEIGHT + 1)) / 100.0;
                    }
                } catch (...) {
                    continue;
                }
                int ww = (int)llround(w * 100.0);                
                flatPairs.push_back(u);
                flatPairs.push_back(v);
                flatWeights.push_back(ww);
                
                if (isDirected) {
                    flatPairs.push_back(v);
                    flatPairs.push_back(u);
                    flatWeights.push_back(ww);}
                maxNodeId = max(maxNodeId, max(u, v));}
            f.close();
            numEdges = (int)flatWeights.size();
            cout << "[DEBUG] Rank 0: Finished reading file\n";
            cout << "Loaded " << numEdges << " directed edges (symmetrized from input)\n";
            cout << "Max node ID: " << maxNodeId << "\n";
            cout.flush();
        }

        MPI_Bcast(&maxNodeId, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            flatPairs.resize(2 * numEdges);
            flatWeights.resize(numEdges);
        }
        if (numEdges > 0) {
            MPI_Bcast(flatPairs.data(), 2 * numEdges, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(flatWeights.data(), numEdges, MPI_INT, 0, MPI_COMM_WORLD);
        }

        nodesPerRank = (maxNodeId + 1 + size - 1) / size;
        int myStart = rank * nodesPerRank;
        int myEnd   = min((rank + 1) * nodesPerRank, maxNodeId + 1);
        if (rank == 0) {
            cout << "\nGraph Partitioning Across " << size << " MPI Ranks\n";
        }
        cout << "Rank " << rank << ": Managing nodes [" << myStart << ", " << myEnd << ")\n";
        for (int i = 0; i < numEdges; ++i) {
            int u = flatPairs[2*i];
            int v = flatPairs[2*i + 1];
            int w = flatWeights[i];

            if (u >= myStart && u < myEnd) {
                Node &nu = localGraph[u];
                if (nu.id == -1) {
                    nu.id = u; 
                    nu.x = (u % 100) * 10.0; 
                    nu.y = (u / 100) * 10.0; 
                    nu.ownerRank = rank;
                }
                nu.edges.push_back({v, w});
                nu.degree++;
                if (!(v >= myStart && v < myEnd)) {
                    ghostSet.insert(v);}}}
        
        for (int nodeId = myStart; nodeId < myEnd; ++nodeId) {
            if (localGraph.find(nodeId) == localGraph.end()) {
                Node &n = localGraph[nodeId];
                n.id = nodeId;
                n.x = (nodeId % 100) * 10.0;
                n.y = (nodeId / 100) * 10.0;
                n.ownerRank = rank;}}

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            cout << "\nGraph distributed successfully!\n";
            cout << "  Local rank 0 nodes: " << localGraph.size() << "\n\n";
        }
        
        verifyGraphConsistency();
    }

    void verifyGraphConsistency() {
        if (rank == 0) {
            cout << "\n[VERIFICATION] Checking graph consistency...\n";
        }
        
        vector<int> allNodeCounts(size);
        int myNodeCount = localGraph.size();
        MPI_Gather(&myNodeCount, 1, MPI_INT, allNodeCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);        
        if (rank == 0) {
            int totalNodes = accumulate(allNodeCounts.begin(), allNodeCounts.end(), 0);
            cout << "Total nodes across all ranks: " << totalNodes << "\n";
            for (int r = 0; r < size; r++) {
                cout << "    Rank " << r << ": " << allNodeCounts[r] << " nodes\n";
            }
        }
        
        int localSymmetryErrors = 0;
        for (auto& [nodeId, node] : localGraph) {
            for (auto& edge : node.edges) {
                int targetOwner = owner_of(edge.to, nodesPerRank, size);
                if (targetOwner == rank) {
                    auto it = localGraph.find(edge.to);
                    if (it != localGraph.end()) {
                        bool found = false;
                        for (auto& reverseEdge : it->second.edges) {
                            if (reverseEdge.to == nodeId && reverseEdge.w == edge.w) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            localSymmetryErrors++;}}}}}
        
        int globalSymmetryErrors = 0;
        MPI_Reduce(&localSymmetryErrors, &globalSymmetryErrors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (globalSymmetryErrors == 0) {
                cout << "Edge symmetry verified (undirected graph)\n";
            } else {
                cout << "Warning: " << globalSymmetryErrors << " symmetry issues found\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            cout << "[VERIFICATION] Graph consistency check complete\n\n";
        }
    }

    void analyzeGraph() {
        if (rank == 0) {
            cout << "\n\nStep 2: Graph Analysis (Distributed)\n";
        }
        long long localDeg = 0;
        int localNodes = (int)localGraph.size();
        for (auto &p : localGraph) localDeg += p.second.degree;
        long long globalDeg = 0;
        int globalNodes = 0;
        MPI_Reduce(&localDeg, &globalDeg, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localNodes, &globalNodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            avgDegree   = globalNodes ? (double)globalDeg / globalNodes : 0.0;
            isRoadNetwork = (avgDegree >= 2.0 && avgDegree <= 4.0);
            isScaleFree   = (avgDegree > 10.0);
            cout << "Global Statistics:\n";
            cout << "  Total nodes: " << globalNodes << "\n";
            cout << "  Average degree: " << fixed << setprecision(2) << avgDegree << "\n";
            cout << "  Graph type: " << (isRoadNetwork ? "Road Network" : (isScaleFree ? "Scale-Free" : "General")) << "\n";
        }
        MPI_Bcast(&avgDegree, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&isRoadNetwork, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(&isScaleFree, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    vector<L1Cluster> buildLocalL1() {
        vector<L1Cluster> localL1;
        unordered_set<int> unassigned;
        for (auto &kv : localGraph) unassigned.insert(kv.first);

        int clusterSize = isRoadNetwork ? 50 : (isScaleFree ? 30 : 20);
        int cid = rank * 1000000;

        while (!unassigned.empty()) {
            int seed = *unassigned.begin();
            L1Cluster c;
            c.id = cid++;
            c.representative = seed;
            c.ownerRank = rank;

            queue<int> q;
            unordered_set<int> seen;
            q.push(seed); seen.insert(seed);
            c.nodes.push_back(seed);
            unassigned.erase(seed);

            while (!q.empty() && (int)c.nodes.size() < clusterSize) {
                int u = q.front(); q.pop();
                auto it = localGraph.find(u);
                if (it == localGraph.end()) continue;
                for (const auto &e : it->second.edges) {
                    int v = e.to;
                    if (seen.count(v)) continue;
                    if (localGraph.find(v) != localGraph.end() && unassigned.count(v)) {
                        seen.insert(v);
                        q.push(v);
                        c.nodes.push_back(v);
                        unassigned.erase(v);
                    }
                    if (localGraph.find(v) == localGraph.end()) c.isBoundary = true;}}
            
            double sx=0, sy=0;
            for (int nodeId : c.nodes) {
                sx += localGraph[nodeId].x; 
                sy += localGraph[nodeId].y;
            }
            if (!c.nodes.empty()) { 
                c.centerX = sx / c.nodes.size(); 
                c.centerY = sy / c.nodes.size(); 
            }
            localL1.push_back(std::move(c));
        }

        cout << "Rank " << rank << ": Built " << localL1.size() << " L1 clusters\n";
        return localL1;
    }

    void gatherGlobalL1_and_broadcast(const vector<L1Cluster>& myL1) {
        const int CHUNK_SIZE = 100;        
        int myCount = (int)myL1.size();
        vector<int> allCounts(this->size, 0);
        MPI_Gather(&myCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int totalClusters = 0;
        if (rank == 0) {
            for (int c : allCounts) totalClusters += c;
            cout << "\n[DEBUG] Total L1 clusters across all ranks: " << totalClusters << "\n";
            cout.flush();
        }
        MPI_Bcast(&totalClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            allL1.clear();
            allL1.reserve(totalClusters);
        }
        
        int globalId = 0;
        for (int srcRank = 0; srcRank < this->size; ++srcRank) {
            int rankClusters = (rank == 0) ? allCounts[srcRank] : 0;
            MPI_Bcast(&rankClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rankClusters == 0) continue;
            
            for (int chunkStart = 0; chunkStart < rankClusters; chunkStart += CHUNK_SIZE) {
                int chunkEnd = min(chunkStart + CHUNK_SIZE, rankClusters);
                int chunkLen = chunkEnd - chunkStart;
                
                vector<int> chunkBuffer;
                
                if (rank == srcRank) {
                    for (int i = chunkStart; i < chunkEnd; ++i) {
                        const auto& c = myL1[i];
                        chunkBuffer.push_back((int)c.nodes.size());
                        chunkBuffer.push_back(c.representative);
                        chunkBuffer.push_back(c.ownerRank);
                        chunkBuffer.push_back(c.isBoundary ? 1 : 0);
                        chunkBuffer.insert(chunkBuffer.end(), c.nodes.begin(), c.nodes.end());
                        chunkBuffer.push_back((int)(c.centerX * 100.0));
                        chunkBuffer.push_back((int)(c.centerY * 100.0));
                    }
                }
                int chunkSize = (rank == srcRank) ? (int)chunkBuffer.size() : 0;
                MPI_Bcast(&chunkSize, 1, MPI_INT, srcRank, MPI_COMM_WORLD);
                
                if (rank != srcRank) {
                    chunkBuffer.resize(chunkSize);
                }
                
                if (chunkSize > 0) {
                    MPI_Bcast(chunkBuffer.data(), chunkSize, MPI_INT, srcRank, MPI_COMM_WORLD);
                }
                int pos = 0;
                for (int i = 0; i < chunkLen && pos < chunkSize; ++i) {
                    L1Cluster c;
                    c.id = globalId++;
                    
                    int nodeCount = chunkBuffer[pos++];
                    c.representative = chunkBuffer[pos++];
                    c.ownerRank = chunkBuffer[pos++];
                    c.isBoundary = (chunkBuffer[pos++] != 0);
                    
                    c.nodes.assign(chunkBuffer.begin() + pos, chunkBuffer.begin() + pos + nodeCount);
                    pos += nodeCount;
                    
                    c.centerX = chunkBuffer[pos++] / 100.0;
                    c.centerY = chunkBuffer[pos++] / 100.0;
                    
                    if (rank == 0) {
                        allL1.push_back(c);}}}}
        
        if (rank == 0) {
            cout << "\nGathered and globalized " << allL1.size() << " L1 clusters\n";
            cout.flush();
            
            nodeToL1.assign(maxNodeId + 1, -1);
            for (const auto& c : allL1) {
                for (int u : c.nodes) {
                    if (u >= 0 && u <= maxNodeId) {
                        nodeToL1[u] = c.id;}}}}
        
        int L1Count = (rank == 0) ? (int)allL1.size() : 0;
        MPI_Bcast(&L1Count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            allL1.clear();
            allL1.reserve(L1Count);
        }
        
        for (int chunkStart = 0; chunkStart < L1Count; chunkStart += CHUNK_SIZE) {
            int chunkEnd = min(chunkStart + CHUNK_SIZE, L1Count);
            int chunkLen = chunkEnd - chunkStart;
            
            vector<int> chunkBuffer;
            
            if (rank == 0) {
                for (int i = chunkStart; i < chunkEnd; ++i) {
                    const auto& c = allL1[i];
                    chunkBuffer.push_back((int)c.nodes.size());
                    chunkBuffer.push_back(c.representative);
                    chunkBuffer.push_back(c.ownerRank);
                    chunkBuffer.push_back(c.isBoundary ? 1 : 0);
                    chunkBuffer.insert(chunkBuffer.end(), c.nodes.begin(), c.nodes.end());
                    chunkBuffer.push_back((int)(c.centerX * 100.0));
                    chunkBuffer.push_back((int)(c.centerY * 100.0));
                }
            }
            
            int chunkSize = (rank == 0) ? (int)chunkBuffer.size() : 0;
            MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rank != 0) {
                chunkBuffer.resize(chunkSize);
            }
            
            if (chunkSize > 0) {
                MPI_Bcast(chunkBuffer.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);
            }
            
            if (rank != 0) {
                int pos = 0;
                for (int i = 0; i < chunkLen && pos < chunkSize; ++i) {
                    L1Cluster c;
                    c.id = chunkStart + i;
                    
                    int nodeCount = chunkBuffer[pos++];
                    c.representative = chunkBuffer[pos++];
                    c.ownerRank = chunkBuffer[pos++];
                    c.isBoundary = (chunkBuffer[pos++] != 0);
                    
                    c.nodes.assign(chunkBuffer.begin() + pos, chunkBuffer.begin() + pos + nodeCount);
                    pos += nodeCount;
                    
                    c.centerX = chunkBuffer[pos++] / 100.0;
                    c.centerY = chunkBuffer[pos++] / 100.0;
                    
                    allL1.push_back(c);}}}
        
        if (rank != 0) {
            nodeToL1.assign(maxNodeId + 1, -1);
            for (const auto& c : allL1) {
                for (int u : c.nodes) {
                    if (u >= 0 && u <= maxNodeId) {
                        nodeToL1[u] = c.id;}}}}
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "Broadcast complete: All ranks have " << allL1.size() << " L1 clusters\n\n";
            cout.flush();
        }
    }

    void buildL2_and_broadcast() {
        if (rank == 0) {
            cout << "\n\nStep 3A: Building L2 clusters (root)\n";
            cout << "[DEBUG] Starting L2 clustering on " << allL1.size() << " L1 clusters...\n";
            cout.flush();
        }
        if (rank == 0) {
            const int gridSize = 200;
            double minX=1e18, minY=1e18, maxX=-1e18, maxY=-1e18;
            
            for (const auto &c : allL1) {
                minX=min(minX,c.centerX); minY=min(minY,c.centerY);
                maxX=max(maxX,c.centerX); maxY=max(maxY,c.centerY);
            }
            double cellW = (maxX-minX)/gridSize + 1;
            double cellH = (maxY-minY)/gridSize + 1;

            unordered_map<long long, vector<int>> hash;
            auto key = [&](int cx,int cy){ return ( (long long)cy<<32 ) | (unsigned long long)cx; };
            
            for (int i=0;i<(int)allL1.size();++i) {
                int cx = (int)((allL1[i].centerX - minX)/cellW);
                int cy = (int)((allL1[i].centerY - minY)/cellH);
                hash[key(cx,cy)].push_back(i);
            }

            int target = max(10, min(100, (int)allL1.size()/1000));
            unordered_set<int> unassigned;
            for (int i=0;i<(int)allL1.size();++i) unassigned.insert(i);
            int id=0;

            while(!unassigned.empty()){
                int seed = *unassigned.begin(); unassigned.erase(seed);
                L2Cluster L2; L2.id = id++; L2.representative = allL1[seed].representative;
                L2.l1Clusters.push_back(seed);

                int scx = (int)((allL1[seed].centerX - minX)/cellW);
                int scy = (int)((allL1[seed].centerY - minY)/cellH);

                for (int dy=-1; dy<=1 && (int)L2.l1Clusters.size()<target; ++dy){
                    for (int dx=-1; dx<=1 && (int)L2.l1Clusters.size()<target; ++dx){
                        int cx=scx+dx, cy=scy+dy;
                        if (cx<0||cx>=gridSize||cy<0||cy>=gridSize) continue;
                        auto it = hash.find(key(cx,cy));
                        if (it==hash.end()) continue;
                        for (int cand : it->second) {
                            if (unassigned.count(cand)){
                                L2.l1Clusters.push_back(cand);
                                unassigned.erase(cand);
                                if ((int)L2.l1Clusters.size()>=target) break;}}}}
                
                double sx=0, sy=0;
                for (int l1id : L2.l1Clusters){ sx+=allL1[l1id].centerX; sy+=allL1[l1id].centerY; }
                L2.centerX = sx / L2.l1Clusters.size();
                L2.centerY = sy / L2.l1Clusters.size();
                allL2.push_back(std::move(L2));
            }

            cout << "Built " << allL2.size() << " L2 clusters\n";
            cout.flush();
        }
        const int CHUNK_SIZE = 100;
        int L2N = (rank==0) ? (int)allL2.size() : 0;
        MPI_Bcast(&L2N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            allL2.resize(L2N);
        }
        for (int chunkStart = 0; chunkStart < L2N; chunkStart += CHUNK_SIZE) {
            int chunkEnd = min(chunkStart + CHUNK_SIZE, L2N);
            
            vector<int> buffer;
            if (rank == 0) {
                for (int i = chunkStart; i < chunkEnd; ++i) {
                    const auto& c = allL2[i];
                    buffer.push_back((int)c.l1Clusters.size());
                    buffer.push_back(c.representative);
                    buffer.push_back((int)(c.centerX * 100.0));
                    buffer.push_back((int)(c.centerY * 100.0));
                    buffer.insert(buffer.end(), c.l1Clusters.begin(), c.l1Clusters.end());
                }
            }

            int bufSize = (rank==0) ? (int)buffer.size() : 0;
            MPI_Bcast(&bufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rank != 0) {
                buffer.resize(bufSize);
            }
            
            if (bufSize > 0) {
                MPI_Bcast(buffer.data(), bufSize, MPI_INT, 0, MPI_COMM_WORLD);
            }

            if (rank != 0) {
                int pos = 0;
                for (int i = chunkStart; i < chunkEnd; ++i) {
                    int count = buffer[pos++];
                    allL2[i].id = i;
                    allL2[i].representative = buffer[pos++];
                    allL2[i].centerX = buffer[pos++] / 100.0;
                    allL2[i].centerY = buffer[pos++] / 100.0;
                    allL2[i].l1Clusters.assign(buffer.begin() + pos, buffer.begin() + pos + count);
                    pos += count;}}}

        l1ToL2.assign(allL1.size(), -1);
        for (int i = 0; i < (int)allL2.size(); ++i) {
            for (int l1 : allL2[i].l1Clusters) {
                if (l1 >= 0 && l1 < (int)l1ToL2.size()) {
                    l1ToL2[l1] = i;
                }
            }
        }
        if (rank == 0) {
            cout << "L2 broadcast complete\n";
            cout.flush();
        }
    }
    void buildSN_and_broadcast() {
        if (rank == 0) {
            cout << "\n\nStep 3B: Building Supernodes (root)\n";
            cout << "[DEBUG] Starting SN clustering on " << allL2.size() << " L2 clusters...\n";
            cout.flush();
        }
        if (rank == 0) {
            const int gridSize=100;
            double minX=1e18, minY=1e18, maxX=-1e18, maxY=-1e18;
            
            for (const auto &c: allL2){
                minX=min(minX,c.centerX); minY=min(minY,c.centerY);
                maxX=max(maxX,c.centerX); maxY=max(maxY,c.centerY);
            }
            double cellW=(maxX-minX)/gridSize + 1;
            double cellH=(maxY-minY)/gridSize + 1;

            unordered_map<long long, vector<int>> hash;
            auto key=[&](int cx,int cy){ return ((long long)cy<<32) | (unsigned long long)cx; };
            
            for (int i=0;i<(int)allL2.size();++i){
                int cx=(int)((allL2[i].centerX-minX)/cellW);
                int cy=(int)((allL2[i].centerY-minY)/cellH);
                hash[key(cx,cy)].push_back(i);
            }

            int target=max(10, min(50, (int)allL2.size()/100));
            unordered_set<int> unassigned;
            for (int i=0;i<(int)allL2.size();++i) unassigned.insert(i);
            int id=0;
            
            while(!unassigned.empty()){
                int seed=*unassigned.begin(); unassigned.erase(seed);
                SuperNode sn; sn.id=id++; sn.representative=allL2[seed].representative;
                sn.l2Clusters.push_back(seed);

                int scx=(int)((allL2[seed].centerX-minX)/cellW);
                int scy=(int)((allL2[seed].centerY-minY)/cellH);
                
                for (int dy=-2;dy<=2 && (int)sn.l2Clusters.size()<target;++dy){
                    for (int dx=-2;dx<=2 && (int)sn.l2Clusters.size()<target;++dx){
                        int cx=scx+dx, cy=scy+dy;
                        if (cx<0||cx>=gridSize||cy<0||cy>=gridSize) continue;
                        auto it=hash.find(key(cx,cy));
                        if (it==hash.end()) continue;
                        for (int cand: it->second){
                            if (unassigned.count(cand)){
                                sn.l2Clusters.push_back(cand);
                                unassigned.erase(cand);
                                if ((int)sn.l2Clusters.size()>=target) break;}}}}
                
                double sx=0, sy=0;
                for (int l2: sn.l2Clusters){ sx+=allL2[l2].centerX; sy+=allL2[l2].centerY; }
                sn.centerX=sx/sn.l2Clusters.size();
                sn.centerY=sy/sn.l2Clusters.size();
                allSN.push_back(std::move(sn));
            }
            cout << "Built " << allSN.size() << " supernodes\n";
            cout.flush();
        }
        const int CHUNK_SIZE = 100;
        int SNN = (rank==0) ? (int)allSN.size() : 0;
        MPI_Bcast(&SNN, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            allSN.resize(SNN);
        }
        for (int chunkStart = 0; chunkStart < SNN; chunkStart += CHUNK_SIZE) {
            int chunkEnd = min(chunkStart + CHUNK_SIZE, SNN);
            vector<int> buffer;
            if (rank == 0) {
                for (int i = chunkStart; i < chunkEnd; ++i) {
                    const auto& sn = allSN[i];
                    buffer.push_back((int)sn.l2Clusters.size());
                    buffer.push_back(sn.representative);
                    buffer.push_back((int)(sn.centerX * 100.0));
                    buffer.push_back((int)(sn.centerY * 100.0));
                    buffer.insert(buffer.end(), sn.l2Clusters.begin(), sn.l2Clusters.end());
                }
            }

            int bufSize = (rank==0) ? (int)buffer.size() : 0;
            MPI_Bcast(&bufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rank != 0) {
                buffer.resize(bufSize);
            }
            
            if (bufSize > 0) {
                MPI_Bcast(buffer.data(), bufSize, MPI_INT, 0, MPI_COMM_WORLD);
            }

            if (rank != 0) {
                int pos = 0;
                for (int i = chunkStart; i < chunkEnd; ++i) {
                    int count = buffer[pos++];
                    allSN[i].id = i;
                    allSN[i].representative = buffer[pos++];
                    allSN[i].centerX = buffer[pos++] / 100.0;
                    allSN[i].centerY = buffer[pos++] / 100.0;
                    allSN[i].l2Clusters.assign(buffer.begin() + pos, buffer.begin() + pos + count);
                    pos += count;
                }
            }
        }

        l2ToSN.assign(allL2.size(), -1);
        for (int i = 0; i < (int)allSN.size(); ++i) {
            for (int l2 : allSN[i].l2Clusters) {
                if (l2 >= 0 && l2 < (int)l2ToSN.size()) {
                    l2ToSN[l2] = i;
                }
            }
        }

        int K = min(snPrecomputeK, (int)allSN.size());
        
        if (rank == 0) {
            cout << "[DEBUG] Precomputing " << K << "x" << K << " SN distance matrix...\n";
            cout.flush();
            
            snDist.assign(K*K, INF);
            for (int i=0;i<K;++i){
                snDist[i*K+i]=0;
                for (int j=i+1;j<K;++j){
                    double dx = allSN[i].centerX - allSN[j].centerX;
                    double dy = allSN[i].centerY - allSN[j].centerY;
                    int d = (int)llround(sqrt(dx*dx+dy*dy)*10.0);
                    snDist[i*K+j]=snDist[j*K+i]=d;
                }
            }
            cout << "[DEBUG] SN distance matrix complete\n";
            cout.flush();
        } else {
            snDist.assign(K*K, 0);
        }
        
        if (K > 0) {
            MPI_Bcast(snDist.data(), K*K, MPI_INT, 0, MPI_COMM_WORLD);
        }
        snPrecomputeK = K;
        
        if (rank == 0) {
            cout << "SN broadcast complete\n";
            cout.flush();
        }
    }
    int supernode_of_node(int u) const {
        if (u < 0 || u > maxNodeId) return -1;
        int l1 = nodeToL1[u];
        if (l1 < 0 || l1 >= (int)l1ToL2.size()) return -1;
        int l2 = l1ToL2[l1];
        if (l2 < 0 || l2 >= (int)l2ToSN.size()) return -1;
        return l2ToSN[l2];
    }
    
    int approx_distance_by_SN(int s, int t) const {
        int sns = supernode_of_node(s);
        int snt = supernode_of_node(t);
        if (sns < 0 || snt < 0 || sns >= (int)allSN.size() || snt >= (int)allSN.size()) {
            auto getXY = [&](int u, double& x, double& y) {
                int r = owner_of(u, nodesPerRank, size);
                if (r == rank) {
                    auto it = localGraph.find(u);
                    if (it != localGraph.end()) { x = it->second.x; y = it->second.y; return true; }
                }
                x = (u % 100) * 10.0; y = (u / 100) * 10.0; return true;
            };
            double xs, ys, xt, yt;
            getXY(s, xs, ys); getXY(t, xt, yt);
            return (int)llround(sqrt((xs-xt)*(xs-xt)+(ys-yt)*(ys-yt))*10.0);
        }
        if (sns < snPrecomputeK && snt < snPrecomputeK) {
            return snDist[sns * snPrecomputeK + snt];
        }
        double dx = allSN[sns].centerX - allSN[snt].centerX;
        double dy = allSN[sns].centerY - allSN[snt].centerY;
        return (int)llround(sqrt(dx*dx + dy*dy) * 10.0);
    }
    int distributed_sssp(int source, int target, bool verbose = true) {
        int srcOwner = owner_of(source, nodesPerRank, size);
        int tgtOwner = owner_of(target, nodesPerRank, size);
        
        int sourceValid = 0, targetValid = 0;
        if (srcOwner == rank && localGraph.count(source)) sourceValid = 1;
        if (tgtOwner == rank && localGraph.count(target)) targetValid = 1;
        
        int globalSourceValid = 0, globalTargetValid = 0;
        MPI_Allreduce(&sourceValid, &globalSourceValid, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(&targetValid, &globalTargetValid, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        if (rank == 0 && verbose) {
            if (!globalSourceValid) cout << "  Warning: Source " << source << " not found in graph\n";
            if (!globalTargetValid) cout << "  Warning: Target " << target << " not found in graph\n";
        }
        
        if (!globalSourceValid || !globalTargetValid) {
            return INF;
        }
        
        unordered_map<int,int> dist;
        
        auto set_if_better = [&](int u, int nd) -> bool {
            auto it = dist.find(u);
            if (it == dist.end() || nd < it->second) { 
                dist[u] = nd; 
                return true; 
            }
            return false;
        };
        
        using P = pair<int,int>;
        auto cmp = [](const P& a, const P& b) { return a.first > b.first; };
        priority_queue<P, vector<P>, decltype(cmp)> pq(cmp);
        
        if (rank == srcOwner) {
            set_if_better(source, 0);
            pq.push({0, source});
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        int iter = 0, done = 0;
        const int MAX_ITERS = 200000;
        auto t0 = high_resolution_clock::now();
        
        int targetFound = 0;
        
        long long totalMessagesSent = 0;
        long long totalMessagesRecv = 0;
        
        int idleIterations = 0;
        const int MAX_IDLE = 3;
        
        while (!done && ++iter < MAX_ITERS) {
            vector<vector<int>> sendBuf(size);
            vector<int> sendCounts(size, 0);
            
            int localWork = 0;
            int relaxBudget = 50000;
            
            while (!pq.empty() && relaxBudget-- > 0) {
                P top = pq.top();
                int du = top.first;
                int u = top.second;
                pq.pop();
                
                auto distIt = dist.find(u);
                if (distIt == dist.end() || du > distIt->second) continue;
                
                if (u == target && owner_of(u, nodesPerRank, size) == rank) {
                    targetFound = 1;
                }
                
                if (owner_of(u, nodesPerRank, size) != rank) continue;
                
                auto it = localGraph.find(u);
                if (it == localGraph.end()) continue;
                
                localWork = 1;
                
                const auto& edges = it->second.edges;
                
                if (edges.size() > 1000) {
                    vector<tuple<int,int,int>> candidates;
                    candidates.reserve(edges.size());
                    
                    for (const auto& e : edges) {
                        long long ndll = (long long)du + e.w;
                        if (ndll >= INF) continue;
                        int nd = (int)ndll;
                        int v = e.to;
                        
                        auto vDistIt = dist.find(v);
                        if (vDistIt == dist.end() || nd < vDistIt->second) {
                            int own = owner_of(v, nodesPerRank, size);
                            candidates.push_back({v, nd, own});
                        }
                    }
                    
                    vector<vector<pair<int,int>>> threadLocal(omp_get_max_threads());
                    vector<vector<vector<int>>> threadRemote(omp_get_max_threads(), vector<vector<int>>(size));
                    
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < candidates.size(); i++) {
                        int v = get<0>(candidates[i]);
                        int nd = get<1>(candidates[i]);
                        int own = get<2>(candidates[i]);
                        int tid = omp_get_thread_num();
                        
                        if (own == rank) {
                            threadLocal[tid].push_back({nd, v});
                        } else {
                            threadRemote[tid][own].push_back(v);
                            threadRemote[tid][own].push_back(nd);
                        }
                    }
                    
                    for (int tid = 0; tid < omp_get_max_threads(); tid++) {
                        for (auto& upd : threadLocal[tid]) {
                            if (set_if_better(upd.second, upd.first)) {
                                pq.push(upd);
                            }
                        }
                        for (int r = 0; r < size; r++) {
                            sendBuf[r].insert(sendBuf[r].end(), 
                                            threadRemote[tid][r].begin(), 
                                            threadRemote[tid][r].end());
                            sendCounts[r] += threadRemote[tid][r].size();
                        }
                    }
                } else {
                    for (const auto& e : edges) {
                        long long ndll = (long long)du + e.w;
                        if (ndll >= INF) continue;
                        int nd = (int)ndll;
                        int v = e.to;
                        int own = owner_of(v, nodesPerRank, size);
                        
                        auto vDistIt = dist.find(v);
                        bool isImprovement = (vDistIt == dist.end() || nd < vDistIt->second);
                        
                        if (!isImprovement) continue;
                        
                        if (own == rank) {
                            if (set_if_better(v, nd)) {
                                pq.push({nd, v});
                            }
                        } else {
                            sendBuf[own].push_back(v);
                            sendBuf[own].push_back(nd);
                            sendCounts[own] += 2;
                        }
                    }
                }
            }
            
            int globalTargetFound = 0;
            MPI_Allreduce(&targetFound, &globalTargetFound, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            
            int totalSend = accumulate(sendCounts.begin(), sendCounts.end(), 0);
            int globalTotalSend = 0;
            MPI_Allreduce(&totalSend, &globalTotalSend, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            if (globalTotalSend == 0 && !localWork) {
                int localActive = !pq.empty() ? 1 : 0;
                int globalActive = 0;
                MPI_Allreduce(&localActive, &globalActive, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                
                if (!globalActive) {
                    idleIterations++;
                    if (idleIterations >= MAX_IDLE) {
                        done = 1;
                    }
                } else {
                    idleIterations = 0;
                }
                continue;
            }
            
            vector<int> recvCounts(size, 0);
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            
            int totalRecv = accumulate(recvCounts.begin(), recvCounts.end(), 0);
            
            totalMessagesSent += totalSend / 2;
            totalMessagesRecv += totalRecv / 2;
            
            vector<int> recvBuf(totalRecv);
            vector<int> sdispls(size, 0), rdispls(size, 0);
            
            for (int i = 1; i < size; i++) {
                sdispls[i] = sdispls[i-1] + sendCounts[i-1];
                rdispls[i] = rdispls[i-1] + recvCounts[i-1];
            }
            
            vector<int> flatSend;
            if (totalSend > 0) {
                flatSend.reserve(totalSend);
                for (int r = 0; r < size; r++) {
                    flatSend.insert(flatSend.end(), sendBuf[r].begin(), sendBuf[r].end());
                }
            }
            
            if (totalSend > 0 || totalRecv > 0) {
                MPI_Alltoallv(
                    totalSend > 0 ? flatSend.data() : nullptr, sendCounts.data(), sdispls.data(), MPI_INT,
                    totalRecv > 0 ? recvBuf.data() : nullptr, recvCounts.data(), rdispls.data(), MPI_INT,
                    MPI_COMM_WORLD
                );
            }
            
            int receivedUpdates = 0;
            for (int i = 0; i + 1 < totalRecv; i += 2) {
                int v = recvBuf[i];
                int nd = recvBuf[i+1];
                
                if (set_if_better(v, nd)) {
                    receivedUpdates = 1;
                    if (owner_of(v, nodesPerRank, size) == rank) {
                        pq.push({nd, v});}}}
            
            int localActive = (localWork || receivedUpdates || !pq.empty()) ? 1 : 0;
            int globalActive = 0;
            MPI_Allreduce(&localActive, &globalActive, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            
            if (!globalActive) {
                idleIterations++;
                if (idleIterations >= MAX_IDLE) {
                    done = 1;
                }
            } else {
                idleIterations = 0;
            }
            
            if (rank == 0 && verbose && (iter == 1 || iter % 20 == 0)) {
                cout << "[DEBUG] Iter " << iter 
                     << " send=" << totalSend/2 
                     << " recv=" << totalRecv/2 
                     << " pqSize=" << pq.size()
                     << " omp=" << omp_get_max_threads()
                     << " target=" << (globalTargetFound ? "FOUND" : "searching") << "\n";
                cout.flush();
            }
        }
        auto t1 = high_resolution_clock::now();
        double sec = duration<double>(t1 - t0).count();
        long long globalMessagesSent = 0, globalMessagesRecv = 0;
        MPI_Reduce(&totalMessagesSent, &globalMessagesSent, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&totalMessagesRecv, &globalMessagesRecv, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int myAns = INF;
        auto it = dist.find(target);
        if (it != dist.end()) {
            myAns = it->second;
        }
        
        int globalMin = INF;
        MPI_Allreduce(&myAns, &globalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        vector<int> allDistances(rank == 0 ? size : 0, INF);
        MPI_Gather(&myAns, 1, MPI_INT, 
                   rank == 0 ? allDistances.data() : nullptr, 
                   1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && verbose) {
            cout << "\n[DEBUG] Distance to target " << target << " on each rank:\n";
            for (int r = 0; r < size; r++) {
                cout << "  Rank " << r << ": " << (allDistances[r] == INF ? "INF" : to_string(allDistances[r])) << "\n";
            }
        }
        
        if (rank == 0 && verbose) {
            cout << "\nResults Summary -\n";
            if (globalMin == INF) 
                cout << "  Distance: UNREACHABLE\n";
            else 
                cout << "  Distance: " << globalMin << "\n";
            cout << "  Total Iterations: " << iter << "\n";
            cout << "  Wall Time: " << fixed << setprecision(6) << sec << " s\n";
            cout << "  Messages Sent: " << globalMessagesSent << "\n";
            cout << "  Messages Received: " << globalMessagesRecv << "\n";
            cout << "  Avg Messages/Iter: " << fixed << setprecision(1) 
                 << (iter > 0 ? (double)globalMessagesSent / iter : 0) << "\n";
        }
        return globalMin;
    }
    int runSequentialDijkstra(int src, int tgt) {
        unordered_map<int, vector<Edge>> graph;
        ifstream f(filename);
        if (!f.is_open()) return INF;
        
        srand(42);
        const int MIN_WEIGHT = 10;
        const int MAX_WEIGHT = 1000;
        
        bool isDirected = false;
        unordered_map<long long, bool> edgeCheck;
        int checkCount = 0;
        const int CHECK_LIMIT = 1000;
        
        string line;
        vector<string> allLines;
        
        while (getline(f, line)) {
            if (!line.empty() && line[0] != '#') {
                allLines.push_back(line);
                if (checkCount < CHECK_LIMIT) {
                    istringstream iss(line);
                    string s; iss >> s;
                    if (s == "source" || s == "Source" || s == "FromNodeId" || s == "ToNodeId") continue;
                    try {
                        int u = stoi(s);
                        int v;
                        if (iss >> v) {
                            long long edgeKey = ((long long)min(u,v) << 32) | (long long)max(u,v);
                            if (edgeCheck.count(edgeKey)) {
                                isDirected = false;
                            } else {
                                edgeCheck[edgeKey] = true;
                            }
                            checkCount++;
                        }
                    } catch (...) {
                        continue;}}}}
        if (checkCount > 50 && edgeCheck.size() == checkCount) {
            isDirected = true;
        }
        
        for (const auto& line : allLines) {
            if (line.empty()) continue;
            istringstream iss(line);
            int u, v;
            double w = 1.0;
            string s;
            iss >> s;
            
            if (s == "source" || s == "Source" || s == "FromNodeId" || s == "ToNodeId") continue;
            try {
                u = stoi(s);
                if (!(iss >> v)) continue;
                
                if (iss >> w) {
                    w = w;
                } else {
                    w = (MIN_WEIGHT + rand() % (MAX_WEIGHT - MIN_WEIGHT + 1)) / 100.0;
                }
            } catch (...) {
                continue;
            }
            int ww = (int)llround(w * 100.0);
            graph[u].push_back({v, ww});
            
            if (isDirected) {
                graph[v].push_back({u, ww});
            }
        }
        f.close();
        unordered_map<int, int> dist;
        auto cmp = [](auto& a, auto& b) { return a.second > b.second; };
        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
        dist[src] = 0;
        pq.push({src, 0});
        while (!pq.empty()) {
            auto [u, du] = pq.top();
            pq.pop();
            if (du != dist[u]) continue;
            if (u == tgt) break;
            for (auto& e : graph[u]) {
                int nd = du + e.w;
                if (!dist.count(e.to) || nd < dist[e.to]) {
                    dist[e.to] = nd;
                    pq.push({e.to, nd});
                }
            }
        }
        
        return dist.count(tgt) ? dist[tgt] : INF;
    }

    void runBatchVerification(int numTests = 10) {
        if (rank == 0) {
            cout << "\n\n";
            cout << "Batch Verification Mode - \n";
            cout << "Running " << numTests << " verification tests...\n\n";
        }
        vector<int> testNodes;
        for (auto& [nodeId, node] : localGraph) {
            if (!node.edges.empty()) {
                testNodes.push_back(nodeId);
            }
        }
        
        int myCount = min(3, (int)testNodes.size());
        vector<int> myTestNodes;
        for (int i = 0; i < myCount; i++) {
            myTestNodes.push_back(testNodes[i]);
        }
        
        vector<int> allCounts(size);
        MPI_Gather(&myCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<int> allTestNodes;
        if (rank == 0) {
            int total = accumulate(allCounts.begin(), allCounts.end(), 0);
            allTestNodes.resize(total);
        }
        
        vector<int> displs(size, 0);
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + allCounts[i-1];
        }
        
        MPI_Gatherv(myTestNodes.data(), myCount, MPI_INT,
                    rank == 0 ? allTestNodes.data() : nullptr,
                    allCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<pair<int,int>> testPairs;
        if (rank == 0 && allTestNodes.size() >= 2) {
            for (int i = 0; i < numTests && i < (int)allTestNodes.size() - 1; i++) {
                int src = allTestNodes[i];
                int tgt = allTestNodes[(i + 1) % allTestNodes.size()];
                testPairs.push_back({src, tgt});
            }
            cout << "Generated " << testPairs.size() << " test pairs\n\n";
        }
        
        int pairCount = rank == 0 ? (int)testPairs.size() : 0;
        MPI_Bcast(&pairCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        verifyStats = VerificationStats();
        
        for (int i = 0; i < pairCount; i++) {
            int src = -1, tgt = -1;
            if (rank == 0) {
                src = testPairs[i].first;
                tgt = testPairs[i].second;
                cout << "Test " << (i+1) << "/" << pairCount << ": " << src << " → " << tgt << "\n";
            }
            
            MPI_Bcast(&src, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&tgt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            bool verboseTest = (i == 0);
            int distDist = distributed_sssp(src, tgt, verboseTest);
            
            if (rank == 0) {
                int distSeq = runSequentialDijkstra(src, tgt);
                
                verifyStats.totalQueries++;
                
                if (distDist == distSeq) {
                    verifyStats.matchedQueries++;
                    cout << "  MATCH: dist=" << (distDist == INF ? "INF" : to_string(distDist)) << "\n";
                } else {
                    verifyStats.mismatchedQueries++;
                    verifyStats.mismatches.push_back({src, tgt, distDist, distSeq});
                    cout << "  MISMATCH: distributed=" << (distDist == INF ? "INF" : to_string(distDist))
                         << " sequential=" << (distSeq == INF ? "INF" : to_string(distSeq)) << "\n";
                    
                    if (distDist != INF && distSeq != INF) {
                        double err = abs(distDist - distSeq) * 100.0 / distSeq;
                        verifyStats.maxError = max(verifyStats.maxError, err);
                        verifyStats.avgError += err;
                    }
                }
                cout << "\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0 && verifyStats.totalQueries > 0) {
            cout << "\n";
            cout << "Verification Results Summary:\n";
            cout << "Total Tests: " << verifyStats.totalQueries << "\n";
            cout << "Matches: " << verifyStats.matchedQueries << " (" 
                 << fixed << setprecision(1) 
                 << (100.0 * verifyStats.matchedQueries / verifyStats.totalQueries) << "%)\n";
            cout << "Mismatches: " << verifyStats.mismatchedQueries << " ("
                 << fixed << setprecision(1)
                 << (100.0 * verifyStats.mismatchedQueries / verifyStats.totalQueries) << "%)\n";
            if (verifyStats.mismatchedQueries > 0) {
                cout << "\nMax Error: " << fixed << setprecision(2) << verifyStats.maxError << "%\n";
                cout << "Avg Error: " << fixed << setprecision(2) 
                     << (verifyStats.avgError / verifyStats.mismatchedQueries) << "%\n";
                cout << "\nMismatch Details:\n";
                cout << "  Source → Target: Distributed / Sequential\n";
                cout << "  " << string(50, '-') << "\n";
                for (auto& [s, t, dd, ds] : verifyStats.mismatches) {
                    cout << "  " << s << " → " << t << ": " 
                         << (dd == INF ? "INF" : to_string(dd)) << " / "
                         << (ds == INF ? "INF" : to_string(ds)) << "\n";
                }
            }
            if (verifyStats.matchedQueries == verifyStats.totalQueries) {
                cout << "\nAll Tests Passed. Distributed algorithm is correct.\n";
            } else {
                cout << "\nSome tests failed. Review mismatches above.\n";
            }
            cout << "\n";}}

    DistanceResult queryDistanceDistributed(int source, int target, bool showApprox = true, bool verify = false) {
        if (rank == 0 && showApprox) {
            cout << "\n\n";
            cout << "Query: Shortest Path from " << source << " to " << target << "\n";
            cout << "[Phase 1: Supernode-level approximation - SYNCHRONOUS]\n";
        }
        auto t0 = high_resolution_clock::now();
        int approx = approx_distance_by_SN(source, target);
        auto t1 = high_resolution_clock::now();
        double tApprox = duration_cast<microseconds>(t1 - t0).count() / 1e6;
        if (rank == 0 && showApprox) {
            cout << "  Approximate distance: ~" << approx << "\n";
            cout << "  Computation time: " << fixed << setprecision(6) << tApprox << " s\n";
            cout << "  → instant from hierarchy\n\n";
            cout << "[Phase 2: Exact distance computation - DISTRIBUTED ASYNC]\n";
        }
        auto t2 = high_resolution_clock::now();
        int exact = distributed_sssp(source, target);
        auto t3 = high_resolution_clock::now();
        double tExact = duration_cast<milliseconds>(t3 - t2).count() / 1000.0;
        if (verify && rank == 0) {
            cout << "\n[VERIFICATION MODE]\n";
            cout << "Running sequential Dijkstra for verification...\n";
            auto tSeqStart = high_resolution_clock::now();
            int seqDist = runSequentialDijkstra(source, target);
            auto tSeqEnd = high_resolution_clock::now();
            double tSeq = duration_cast<milliseconds>(tSeqEnd - tSeqStart).count() / 1000.0;
            
            cout << "Sequential distance: " << (seqDist == INF ? "INF" : to_string(seqDist)) << "\n";
            cout << "Sequential time: " << fixed << setprecision(6) << tSeq << " s\n";
            
            if (exact == seqDist) {
                cout << "Verification Passed: Distances match.\n";
            } else {
                cout << "Verification Failed: Distance mismatch.\n";
                cout << "   Distributed: " << (exact == INF ? "INF" : to_string(exact)) << "\n";
                cout << "   Sequential:  " << (seqDist == INF ? "INF" : to_string(seqDist)) << "\n";
            }
            
            if (tExact > 0 && tSeq > 0) {
                cout << "Speedup: " << fixed << setprecision(2) << (tSeq / tExact) << "x\n";
            }
        }
        if (rank == 0 && showApprox) {
            cout << "\n[APPROXIMATE - Level 3 Supernode]\n";
            if (approx < INF) cout << "  Distance: ~" << approx << " (approximate)\n";
            else cout << "  Distance: UNREACHABLE\n";
            cout << "  Time: " << fixed << setprecision(6) << tApprox << " s\n";
            cout << "  Processing: SYNCHRONOUS on root\n\n";

            cout << "[EXACT - Level 0 Full Resolution]\n";
            if (exact < INF) cout << "  Distance: " << exact << " (exact)\n";
            else cout << "  Distance: UNREACHABLE\n";
            cout << "  Time: " << fixed << setprecision(6) << tExact << " s\n";
            cout << "  Processing: ASYNCHRONOUS distributed\n\n";

            cout << "[TOTAL QUERY TIME]\n";
            cout << "  " << fixed << setprecision(6) << (tApprox + tExact) << " s\n";

            if (approx < INF && exact < INF && exact > 0) {
                double err = fabs((double)exact - (double)approx) * 100.0 / (double)exact;
                cout << "\n[APPROXIMATION QUALITY]\n";
                cout << "  Error: " << fixed << setprecision(2) << err << "%\n";
            }
            cout << "\n";
        }
        DistanceResult res;
        res.distance = exact;
        res.isApproximate = false;
        res.hierarchyLevel = 0;
        res.computationTime = tApprox + tExact;
        return res;}
    void printStatistics() {
        if (rank == 0) {
            cout << "\n\n";
            cout << "Alogrithm Statistics:\n";            
            cout << "Hierarchy Structure:\n";
            cout << "  Level 3 (Supernodes): " << allSN.size() << "\n";
            cout << "  Level 2 (L2 Clusters): " << allL2.size() << "\n";
            cout << "  Level 1 (L1 Clusters): " << allL1.size() << "\n";
            cout << "  Level 0 (Nodes): Distributed across " << size << " ranks\n\n";
            cout << "Graph Characteristics:\n";
            cout << "  Type: " << (isRoadNetwork ? "Road Network" : (isScaleFree ? "Scale-Free" : "General")) << "\n";
            cout << "  Average degree: " << fixed << setprecision(2) << avgDegree << "\n\n";
            cout << "Distribution:\n";
            cout << "  MPI Ranks: " << size << "\n";
            cout << "  Processing: Truly Distributed (owner-push SSSP)\n";
            if (verifyStats.totalQueries > 0) {
                cout << "\nVerification Statistics:\n";
                cout << "  Total Queries Verified: " << verifyStats.totalQueries << "\n";
                cout << "  Success Rate: " << fixed << setprecision(1) 
                     << (100.0 * verifyStats.matchedQueries / verifyStats.totalQueries) << "%\n";}}}};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool verifyMode = false;
    bool batchVerifyMode = false;
    int batchTests = 10;
    if (rank == 0) {
        for (int i = 1; i < argc; i++) {
            if (string(argv[i]) == "--verify") {
                verifyMode = true;
            } else if (string(argv[i]) == "--batch-verify") {
                batchVerifyMode = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    batchTests = atoi(argv[i+1]);
                }
            }
        }
    }
    MPI_Bcast(&verifyMode, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&batchVerifyMode, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&batchTests, 1, MPI_INT, 0, MPI_COMM_WORLD);
    HADSSSP had(rank, size);
    vector<string> tryFiles = {
        "roadNet-CA.txt",
        "web-Google.txt",
        "higgs-social_network.txt",
        "graph500-scale20-ef16_adjedges.txt",
        "graph500-scale22-ef16_adjedges.txt",
        "test_small.txt",
        "random_citation_graph.txt",
        "random_citation_graph.csv"};
    string filename;
    int fileFound = 0;
    if (rank == 0) {
        for (auto &f : tryFiles) {
            ifstream t(f);
            if (t.good()) { 
                filename = f; 
                fileFound = 1;
                t.close(); 
                break; 
            }
            t.close();
        }
        if (!fileFound) {
            cerr << "\nERROR: No graph dataset found!\n";
            cerr << "Please create one of these files:\n";
            for (auto &f : tryFiles) cerr << "  - " << f << "\n";
        }
    } 
    MPI_Bcast(&fileFound, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (!fileFound) {
        MPI_Finalize();
        return 1;
    }
    int flen = (rank == 0) ? (int)filename.size() : 0;
    MPI_Bcast(&flen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) filename.resize(flen);
    if (flen > 0) MPI_Bcast(&filename[0], flen, MPI_CHAR, 0, MPI_COMM_WORLD);

    had.loadAndDistributeGraph(filename);
    had.analyzeGraph();

    auto myL1 = had.buildLocalL1();
    had.gatherGlobalL1_and_broadcast(myL1);

    had.buildL2_and_broadcast();
    had.buildSN_and_broadcast();

    if (batchVerifyMode) {
        had.runBatchVerification(batchTests);
        had.printStatistics();
        
        if (rank == 0) cout << "\n\nHAD-SSSP Execution Completed.\n\n";
        MPI_Finalize();
        return 0;}
    vector<pair<int,int>> demoPairs;
    if (rank == 0) {
        vector<int> locals;
        for (auto& kv : had.getLocalGraph()) {
            locals.push_back(kv.first);
        }
        sort(locals.begin(), locals.end());
        
        if (locals.size() >= 4) {
            for (int i = 0; i < min(3, (int)locals.size()); i++) {
                int src = locals[i];
                const auto& node = had.getLocalGraph().find(src)->second;
                if (!node.edges.empty()) {
                    int tgt = node.edges.front().to;
                    demoPairs.push_back({src, tgt});}}}
        cout << "\n\nINTERACTIVE QUERY MODE\n";
        cout << "[AUTO-DEMO] Running " << demoPairs.size() << " example queries.\n\n";
    }
    int numPairs = (rank == 0) ? (int)demoPairs.size() : 0;
    MPI_Bcast(&numPairs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int qi = 0; qi < numPairs; ++qi) {
        int source = -1, target = -1;
        if (rank == 0) {
            source = demoPairs[qi].first;
            target = demoPairs[qi].second;
            cout << "Query " << qi+1 << ": Shortest path from " << source << " to " << target << "\n";
        }
        MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
        had.queryDistanceDistributed(source, target, true, verifyMode);
        MPI_Barrier(MPI_COMM_WORLD);}
    if (rank == 0) {
        cout << "\n\nInteractive Query Mode:\n";
        cout << "Enter custom queries or type 'quit' to exit\n";
        cout << "Format: <source> <target>\n";
        if (verifyMode) {
            cout << "Verification mode: ON (each query will be verified)\n";}}
    int continueLoop = 1;
    while (continueLoop) {
        int source = -1, target = -1;
        if (rank == 0) {
            cout << "\nEnter source and target nodes (or 'quit'): ";
            cout.flush();
            string line;
            if (!getline(cin, line)) {
                continueLoop = 0;
            } else if (line == "quit" || line == "q" || line == "exit") {
                continueLoop = 0;
            } else {
                istringstream iss(line);
                if (iss >> source >> target) {
                } else {
                    cout << "Invalid input. Please enter two node IDs.\n";
                    source = -1;
                    target = -1;}}}
        MPI_Bcast(&continueLoop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!continueLoop) break;
        MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (source >= 0 && target >= 0) {
            if (rank == 0) {
                cout << "\nProcessing query: " << source << " → " << target << "\n";}
            had.queryDistanceDistributed(source, target, true, verifyMode);}
        MPI_Barrier(MPI_COMM_WORLD);}
    had.printStatistics();
    if (rank == 0) cout << "\n\nHAD-SSSP Execution Completed.\n\n";
    MPI_Finalize();
    return 0;}