#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>

/*
compile: g++ -o topk TopK.cu
run    : ./topk
*/

using namespace std;

const int VECTOR_DIM = 50;
const int NUM_VECTORS = 100;
const int TOPK = 10;

/*
  This method has to be implemented sequentially and in CUDA so that you can verify your answers
  Query vector is the first vector in the input array "hostVectors"". Each vector is of the same size = VECTOR_DIM. 
  Store the Jaccard distances of input vectors from the query vector in "answers" array
*/
void Jaccard_Similarity(double *hostVectors, double *queryVec, double *answers, int VECTOR_DIM, int NUM_VECTORS ) {
    for (int i = 0; i < NUM_VECTORS; i++) {
        double sumMin = 0.0;
        double sumMax = 0.0;
        for (int j = 0; j < VECTOR_DIM; j++) {
            double xi = hostVectors[i * VECTOR_DIM + j];
            double yi = queryVec[j];
            sumMin += min(xi, yi);
            sumMax += max(xi, yi);
        }
        double jaccardSimilarity;
        if (sumMax == 0) {
            jaccardSimilarity = 0;
        } else {
            jaccardSimilarity = sumMin / sumMax;
        }
        answers[i] = jaccardSimilarity;
    }
}

__global__ void jaccardKernel(double *hostVectors, double *queryVec, double *answers, int VECTOR_DIM, int NUM_VECTORS) {
    //calculating the thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //checking if the thread us within the bounds of the vectors
    if (i < NUM_VECTORS) {
        double sumMin = 0.0;
        double sumMax = 0.0;
        //looping through vectors to calculate weighted Jaccard
        for (int j = 0; j < VECTOR_DIM; j++) {
            double xi = hostVectors[i * VECTOR_DIM + j];
            double yi = queryVec[j];
            sumMin += min(xi, yi);
            sumMax += max(xi, yi);
        }
        // calculating Weighted Jaccard similarity
        double jaccardSimilarity;
        if (sumMax == 0) {
            jaccardSimilarity = 0;
        } else {
            jaccardSimilarity = sumMin / sumMax;
        }
        answers[i] = jaccardSimilarity;
    }
}

void Parallel_Jaccard_Similarity(double *hostVectors, double *queryVec, double *answers, int VECTOR_DIM, int NUM_VECTORS) {
    double *d_hostVectors, *d_queryVec, *d_answers;
    size_t sizeVectors = NUM_VECTORS * VECTOR_DIM * sizeof(double);
    size_t sizeAnswers = NUM_VECTORS * sizeof(double);

    cudaMalloc((void **)&d_hostVectors, sizeVectors);
    cudaMalloc((void **)&d_queryVec, VECTOR_DIM * sizeof(double));
    cudaMalloc((void **)&d_answers, sizeAnswers);

    cudaMemcpy(d_hostVectors, hostVectors, sizeVectors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queryVec, queryVec, VECTOR_DIM * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (NUM_VECTORS + blockSize - 1) / blockSize;
    jaccardKernel<<<numBlocks, blockSize>>>(d_hostVectors, d_queryVec, d_answers, VECTOR_DIM, NUM_VECTORS);

    cudaMemcpy(answers, d_answers, sizeAnswers, cudaMemcpyDeviceToHost);

    cudaFree(d_hostVectors);
    cudaFree(d_queryVec);
    cudaFree(d_answers);
}

/*
  Query vector is the first vector in the input array "hostVectors"". Each vector is of the same size = VECTOR_DIM. 
  Euclidean distance computation is just an example of a distance. 
  I wrote it so that you know how to process the hostVectors and the queryVec.
*/
void euclidean_distanceArr(double *hostVectors, double *queryVec, double *answers, int VECTOR_DIM, int NUM_VECTORS ) 
{
    for( int i = 0; i<NUM_VECTORS; i++)
    {
      double sum = 0.0;
      for(int j = 0; j<VECTOR_DIM; j++)
      { 
        double val = hostVectors[ i * VECTOR_DIM + j ] - queryVec[j];
        sum += val * val; 
      }
      cout<< "dist of vector "<< i <<" to queryVec "<<std::sqrt(sum)<<"\n";
      answers[i] = std::sqrt(sum);
    }

}

void populatehostVectors( vector<std::vector<double> > vectors, double *hostVectors )
{
   int n = vectors.size();
   for( int i = 0; i<n; i++)
   {
     vector<double> vec = vectors[i];
     for(int j = 0; j<vec.size(); j++)
     {
       hostVectors[ i * VECTOR_DIM + j ] = vec[j];
     }
   }
}

// Function to extract top 10 elements in descending order
double* topK(double* arr, int N) {

    // Create a copy of the input array so we don't modify the original
    double* copy = new double[N];
    std::copy(arr, arr + N, copy);

    // Sort in descending order
    std::sort(copy, copy + N, std::greater<double>());

    // Allocate new array for top 10 elements
    double* top = new double[TOPK];
    std::copy(copy, copy + TOPK, top);

    delete[] copy;  // clean up temporary copy
    return top;     // caller must delete[] this later
}

void printResult(double* ans, int topK) {
    if (ans == nullptr || topK <= 0) {
        std::cout << "No results to display.\n";
        return;
    }

    std::cout << "Top " << topK << " elements:\n";
    for (int i = 0; i < topK; ++i) {
        std::cout << ans[i];
        if (i < topK - 1)
            std::cout << ", ";
    }
    std::cout << std::endl;
}

int main() {
    std::ifstream file("random_vectors_100x50.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open random_vectors_100x50.csv\n";
        return 1;
    }

    std::string line;
    std::vector<std::vector<double> > vectors;

    // Parse CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> vec;
        std::string value;
        while (std::getline(ss, value, ',')) {
            vec.push_back(std::stod(value));
        }
        vectors.push_back(vec);
    }
    file.close();

    size_t n = vectors.size();
    std::cout << "Loaded " << n << " vectors of dimension " << vectors[0].size() << ".\n";

    double *hostVectors = (double *)malloc(sizeof(double) * VECTOR_DIM * NUM_VECTORS);
    populatehostVectors(vectors, hostVectors);

    // to store distances between elements of input vector from a query vector
    double seq_answers[NUM_VECTORS];

    euclidean_distanceArr(hostVectors, hostVectors, seq_answers, VECTOR_DIM, NUM_VECTORS);

    Jaccard_Similarity(hostVectors, hostVectors, seq_answers, VECTOR_DIM, NUM_VECTORS);

    double *topK_SimilarItems = topK(seq_answers, NUM_VECTORS);

    // parallel version
    double par_answers[NUM_VECTORS];

    euclidean_distanceArr(hostVectors, hostVectors, par_answers, VECTOR_DIM, NUM_VECTORS);

    Parallel_Jaccard_Similarity(hostVectors, hostVectors, par_answers, VECTOR_DIM, NUM_VECTORS);

    double *par_topK_SimilarItems = topK(par_answers, NUM_VECTORS);

    printResult(par_topK_SimilarItems, TOPK);
    
    std::cout << "Done\n";
    return 0;
}
