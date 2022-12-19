// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <assert.h>
#include "ligra.h"

// #include <Eigen/Dense>
#include "Eigen/Dense"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;

#include <random>
#include <vector>
#include <map>

// #ifdef CILK
#include <cilk/reducer.h>
// #endif
//#include <cilk/reducer_opadd.h>

#include "parse_pubmed.cpp"

double PARAM_ADAM_B1 = 0.9;
double PARAM_ADAM_B2 = 0.999;
double PARAM_ADAM_EPSILON = 1e-8;


/*
	Utility Functions
*/

void zero_init(MatrixXf& mat) {
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      mat(i,j) = 0.0;
    }
  }
}

void random_init(std::default_random_engine& gen, MatrixXf& mat) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0/(mat.rows()*mat.cols()));
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      mat(i,j) = distribution(gen);
    }
  }
}

void apply_gradient_update_ADAM(std::vector<MatrixXf>& weights, std::vector<MatrixXf>& d_weights,
                                std::vector<MatrixXf>& vel, std::vector<MatrixXf>& mom,
                                double mul, double lr, int t) {

  double lr_t = lr * (sqrt(1.0-pow(PARAM_ADAM_B2, t)) / (1.0-pow(PARAM_ADAM_B1, t)));

  parallel_for (int i = 0; i < weights.size(); i++) {
    parallel_for (int j = 0; j < weights[i].rows(); j++) {
      parallel_for (int k = 0; k < weights[i].cols(); k++) {
        double g = d_weights[i](j,k) * mul;
        double m = mom[i](j,k);
        double v = vel[i](j,k);

        double m_t = PARAM_ADAM_B1 * m + (1.0 - PARAM_ADAM_B1) * g;
        double v_t = PARAM_ADAM_B2 * v + (1.0 - PARAM_ADAM_B2) * (g*g);

        double new_val = weights[i](j,k) - lr_t * m_t / (sqrt(v_t) + PARAM_ADAM_EPSILON);

        mom[i](j,k) = m_t;
        vel[i](j,k) = v_t;
        weights[i](j,k) = new_val;
      }
    }
  }
}

/*
	Gradient table reducer
*/

#include "gradient_table_reducer.cpp"


/*
	Functions/Operations plus functions to propagate their adjoints backwards.
*/


void softmax(MatrixXf& input, MatrixXf& output) {
  // max val divided out for numerical stability
  double mval = input.maxCoeff();//maxval(input);
  output = (input.array() - mval).exp()/((exp(input.array() - mval)).sum());
}

void d_softmax(MatrixXf& input, MatrixXf& d_input, MatrixXf& output, MatrixXf& d_output) {
  d_input = -1.0*output.array() * (d_output.array()*output.array()).sum() + output.array()*d_output.array();
  zero_init(d_output);
}

void crossentropy(MatrixXf& yhat, MatrixXf& y, double& output) {
  double loss_sum = 0.0;
  double n = y.rows()*y.cols();
  for (int i = 0; i < y.rows(); i++) {
    for (int j = 0; j < y.cols(); j++) {
      loss_sum += -1.0 * y(i,j)*log(yhat(i,j) + 1e-12) - (1.0-y(i,j))*log(1-yhat(i,j) + 1e-12);
    }
  }
  output = loss_sum / n;
}

void d_crossentropy(MatrixXf& yhat, MatrixXf& d_yhat, MatrixXf& y, double& d_output) {
   double n = y.rows()*y.cols();
   for (int i = 0; i < y.rows(); i++) {
    for (int j = 0; j < y.cols(); j++) {
      d_yhat(i,j) = d_output*(-1.0 * y(i,j) / (yhat(i,j) + 1e-12) + (1.0-y(i,j))*1.0/(1-yhat(i,j)+1e-12)) * (1.0/n);
    }
  }
}

void sqloss(MatrixXf& input1, MatrixXf& input2, double& output) {
  MatrixXf diff = input1-input2;
  output = (diff*diff).sum();
}

void d_sqloss(MatrixXf& input1, MatrixXf& d_input1, MatrixXf& input2,
              double& d_output) {
  d_input1 = 2*(input1-input2) * d_output;
  d_output = 0.0;
}

template <class vertex>
struct GCN_applyweights_F {

  graph<vertex>& GA;

  uintE* Parents;
  MatrixXf& weights;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* next_vertex_embeddings;
  bool first;
  GCN_applyweights_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings) : Parents(_Parents), GA(_GA), weights(_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings) {}

  inline bool operator() (uintE v) {
    int in_degree = GA.V[v].getOutDegree();
    if (in_degree == 0) return 1;
    uintE n = v;
    next_vertex_embeddings[v] = (weights * prev_vertex_embeddings[n]);
    return 1;
  }
};

template <class vertex>
struct d_GCN_applyweights_F {

  graph<vertex>& GA;

  uintE* Parents;

  MatrixXf& weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;

  MatrixXf* d_weights;

  MatrixXf* d_next_vertex_embeddings;
  MatrixXf* d_prev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_applyweights_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _d_weights,
        MatrixXf* _next_vertex_embeddings, MatrixXf* _d_next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _d_prev_vertex_embeddings,
        ArrayReducer* _reducer) :
                                          Parents(_Parents), GA(_GA),
                                          weights(_weights), d_weights(_d_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          d_next_vertex_embeddings(_d_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
                                          reducer(_reducer) {}

  inline bool operator() (uintE v) {
    int in_degree = GA.V[v].getOutDegree();
    if (in_degree == 0) return 1;
    ArrayReducerView* view = reducer->view();
    MatrixXf& d_weights_view = *(view->get_view(d_weights));
    uintE n = v;

    for (int j = 0; j < weights.rows(); j++) {
      if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
      for (int k = 0; k < weights.cols(); k++) {
        d_weights_view(j,k) += prev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
      }
    }

    MatrixXf& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));

    for (int j = 0; j < weights.rows(); j++) {
      if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
      for (int k = 0; k < weights.cols(); k++) {
        d_prev_vertex_embeddings_n(k,0) += weights(j,k) * d_next_vertex_embeddings[v](j,0);
      }
    }
    return 1;
  }
};



template <class vertex>
struct GCN_edgeMap_F {

  graph<vertex>& GA;

  uintE* Parents;
  MatrixXf& weights;
  MatrixXf& skip_weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* prevprev_vertex_embeddings;

  bool first;
  GCN_edgeMap_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf& _skip_weights, MatrixXf* _next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _prevprev_vertex_embeddings, bool _first) :
                                          Parents(_Parents), GA(_GA), weights(_weights),
                                          skip_weights(_skip_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          prevprev_vertex_embeddings(_prevprev_vertex_embeddings),
                                          first(_first) {}

  inline bool update(uintE v, uintE n) {
    int in_degree = GA.V[v].getOutDegree();

    // self edge
    if (v == n) {
      next_vertex_embeddings[v] += skip_weights * prevprev_vertex_embeddings[n];
    }

    //printf("edge seen %d,%d\n", v,n);
    //if (GA.V[v].getOutDegree() != GA.V[v].getInDegree()) printf("Error in and out degrees do not match!\n");
    //if (GA.V[n].getOutDegree() != GA.V[n].getInDegree()) printf("Error in and out degrees do not match!\n");
    //if (in_degree * GA.V[n].getOutDegree() == 0) printf("Error product of degrees is zero!\n");


    //if (in_degree == 0) printf("error in_degree is zero somehow...\n");
    //if (GA.V[n].getOutDegree() == 0) printf("error GA.V[n].getOutDegree is zero somehow... %d %d i=%d n=%d\n", GA.V[n].getOutDegree(), GA.V[n].getInDegree(), i, n);

    double edge_weight = 1.0 / sqrt(in_degree * GA.V[n].getOutDegree());
    next_vertex_embeddings[v] += prev_vertex_embeddings[n] * edge_weight;

    //if (!first) {
    //  next_vertex_embeddings[v] = next_vertex_embeddings[v].cwiseMax(0.0);
    //}

    return 1;
  }

  inline bool updateAtomic(uintE v, uintE n) {
    return update(v,n);
  }

  inline bool cond(uintE n) {
    return cond_true(n);
  }
};

template <class vertex>
struct d_GCN_edgeMap_F {

  graph<vertex>& GA;

  uintE* Parents;

  MatrixXf& weights;
  MatrixXf& skip_weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* prevprev_vertex_embeddings;
  bool first;

  MatrixXf* d_weights;
  MatrixXf* d_skip_weights;

  MatrixXf* d_next_vertex_embeddings;

  MatrixXf* d_prev_vertex_embeddings;
  MatrixXf* d_prevprev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_edgeMap_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _d_weights,
        MatrixXf& _skip_weights, MatrixXf* _d_skip_weights,
        MatrixXf* _next_vertex_embeddings, MatrixXf* _d_next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _d_prev_vertex_embeddings,
        MatrixXf* _prevprev_vertex_embeddings, MatrixXf* _d_prevprev_vertex_embeddings,
        ArrayReducer* _reducer, bool _first) :
            Parents(_Parents), GA(_GA), weights(_weights), d_weights(_d_weights),
            skip_weights(_skip_weights), d_skip_weights(_d_skip_weights),
            next_vertex_embeddings(_next_vertex_embeddings),
            d_next_vertex_embeddings(_d_next_vertex_embeddings),
            prev_vertex_embeddings(_prev_vertex_embeddings),
            d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
            prevprev_vertex_embeddings(_prevprev_vertex_embeddings),
            d_prevprev_vertex_embeddings(_d_prevprev_vertex_embeddings), reducer(_reducer),
            first(_first) {}

  inline bool update(uintE v, uintE n) {
    int in_degree = GA.V[v].getOutDegree();

    // reverse-mode of fmax(0.0, next_vertex_embeddings[v]).
    //if (!first) {
    //  for (int i = 0; i < next_vertex_embeddings[v].rows(); i++) {
    //    if (next_vertex_embeddings[v](i,0) <= 0.0) {
    //      d_next_vertex_embeddings[v](i,0) = 0.0;
    //    }
    //  }
    //}
    ArrayReducerView* view = reducer->view();
    MatrixXf& d_skip_weights_view = *(view->get_view(d_skip_weights));

    // self edge.
    if (v == n) {
      //return 1;
      for (int j = 0; j < weights.rows(); j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.cols(); k++) {
          d_skip_weights_view(j,k) += prevprev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
        }
      }
      MatrixXf& d_prevprev_vertex_embeddings_n = *(view->get_view(&(d_prevprev_vertex_embeddings[n])));
      for (int j = 0; j < weights.rows(); j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.cols(); k++) {
          d_prevprev_vertex_embeddings_n(k,0) += skip_weights(j,k) * d_next_vertex_embeddings[v](j,0);
        }
      }
    }

    //printf("edge map %d,%d\n", v, n);
    // reverse of matrix multiplies.
    double edge_weight = 1.0/sqrt(in_degree * GA.V[n].getOutDegree());

    MatrixXf& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));
    // propagate to d_prev_vertex_embeddings[n]
    d_prev_vertex_embeddings_n += d_next_vertex_embeddings[v]*edge_weight;
    return 1;
  }
  inline bool updateAtomic(uintE v, uintE n) {
    return update(v,n);
  }

  inline bool cond(uintE n) {
    return cond_true(n);
  }


};






template <class vertex>
struct GCN_F {

  graph<vertex>& GA;

  uintE* Parents;
  MatrixXf& weights;
  MatrixXf& skip_weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* prevprev_vertex_embeddings;

  bool first;
  GCN_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf& _skip_weights, MatrixXf* _next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _prevprev_vertex_embeddings, bool _first) :
                                          Parents(_Parents), GA(_GA), weights(_weights),
                                          skip_weights(_skip_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          prevprev_vertex_embeddings(_prevprev_vertex_embeddings),
                                          first(_first) {}

  inline bool operator() (uintE v) {
    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getOutNeighbors());
    int in_degree = GA.V[v].getOutDegree();

    // self edge
    {
      uintE n = v;
      next_vertex_embeddings[v] = skip_weights * prevprev_vertex_embeddings[n];
    }
    //if (GA.V[v].getOutDegree() != GA.V[v].getInDegree()) printf("Error in and out degrees do not match!\n");
    if (in_degree > 0) {
      for (int i = 0; i < in_degree; i++) {
        uintE n = neighbors[i];
        //if (GA.V[n].getOutDegree() != GA.V[n].getInDegree()) printf("Error in and out degrees do not match!\n");
        //if (in_degree * GA.V[n].getOutDegree() == 0) printf("Error product of degrees is zero!\n");


        //if (in_degree == 0) printf("error in_degree is zero somehow...\n");
        //if (GA.V[n].getOutDegree() == 0) printf("error GA.V[n].getOutDegree is zero somehow... %d %d i=%d n=%d\n", GA.V[n].getOutDegree(), GA.V[n].getInDegree(), i, n);

        double edge_weight = 1.0 / sqrt(in_degree * GA.V[n].getOutDegree());
        next_vertex_embeddings[v] += prev_vertex_embeddings[n] * edge_weight;
      }
    }

    if (!first) {
      next_vertex_embeddings[v] = next_vertex_embeddings[v].cwiseMax(0.0);
    }

    return 1;
  }
};

template <class vertex>
struct d_GCN_F {

  graph<vertex>& GA;

  uintE* Parents;

  MatrixXf& weights;
  MatrixXf& skip_weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* prevprev_vertex_embeddings;
  bool first;

  MatrixXf* d_weights;
  MatrixXf* d_skip_weights;

  MatrixXf* d_next_vertex_embeddings;

  MatrixXf* d_prev_vertex_embeddings;
  MatrixXf* d_prevprev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _d_weights,
        MatrixXf& _skip_weights, MatrixXf* _d_skip_weights,
        MatrixXf* _next_vertex_embeddings, MatrixXf* _d_next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _d_prev_vertex_embeddings,
        MatrixXf* _prevprev_vertex_embeddings, MatrixXf* _d_prevprev_vertex_embeddings,
        ArrayReducer* _reducer, bool _first) :
            Parents(_Parents), GA(_GA), weights(_weights), d_weights(_d_weights),
            skip_weights(_skip_weights), d_skip_weights(_d_skip_weights),
            next_vertex_embeddings(_next_vertex_embeddings),
            d_next_vertex_embeddings(_d_next_vertex_embeddings),
            prev_vertex_embeddings(_prev_vertex_embeddings),
            d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
            prevprev_vertex_embeddings(_prevprev_vertex_embeddings),
            d_prevprev_vertex_embeddings(_d_prevprev_vertex_embeddings), reducer(_reducer),
            first(_first) {}

  inline bool operator() (uintE v) {
    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getOutNeighbors());
    int in_degree = GA.V[v].getOutDegree();

    // reverse-mode of fmax(0.0, next_vertex_embeddings[v]).
    if (!first) {
      for (int i = 0; i < next_vertex_embeddings[v].rows(); i++) {
        if (next_vertex_embeddings[v](i,0) <= 0.0) {
          d_next_vertex_embeddings[v](i,0) = 0.0;
        }
      }
    }

    ArrayReducerView* view = reducer->view();
    MatrixXf& d_skip_weights_view = *(view->get_view(d_skip_weights));

    // self edge.
    {
      uintE n = v;
      for (int j = 0; j < weights.rows(); j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.cols(); k++) {
          d_skip_weights_view(j,k) += prevprev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
        }
      }
      MatrixXf& d_prevprev_vertex_embeddings_n = *(view->get_view(&(d_prevprev_vertex_embeddings[n])));
      for (int j = 0; j < weights.rows(); j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.cols(); k++) {
          d_prevprev_vertex_embeddings_n(k,0) += skip_weights(j,k) * d_next_vertex_embeddings[v](j,0);
        }
      }
    }

    // reverse of matrix multiplies.
    for (int i = 0; i < in_degree; i++) {
      uintE n = neighbors[i];
      double edge_weight = 1.0/sqrt(in_degree * GA.V[n].getOutDegree());

      MatrixXf& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));
      // propagate to d_prev_vertex_embeddings[n]
      d_prev_vertex_embeddings_n += d_next_vertex_embeddings[v]*edge_weight;
    }
    return 1;
  }
};


/*
	Main compute function.
*/

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) {
  long start = P.getOptionLongValue("-r",0);
  long n = GA.n;
  //creates Parents array, initialized to all -1, except for start
  uintE* Parents = newA(uintE,n);
  parallel_for(long i=0;i<n;i++) Parents[i] = UINT_E_MAX;
  Parents[start] = start;

  bool* vIndices = static_cast<bool*>(malloc(sizeof(bool)*GA.n));

  int n_vertices = GA.n;
  int feature_dim = 500;
  bool* is_train = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));
  bool* is_val = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));
  bool* is_test = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));

  std::vector<MatrixXf> groundtruth_labels;
  std::vector<MatrixXf> feature_vectors;

  for (int i = 0; i < n_vertices; i++) {
    MatrixXf tmp(3,1);
    zero_init(tmp);
    groundtruth_labels.push_back(tmp);

    MatrixXf tmp2(feature_dim,1);
    zero_init(tmp2);
    feature_vectors.push_back(tmp2);
  }

  // parse the data from the graph.
  parse_pubmed_data("datasets/pubmed.trainlabels", "datasets/pubmed.vallabels",
                    "datasets/pubmed.testlabels", "datasets/pubmed_features", is_train,
                    is_val, is_test, groundtruth_labels,
                    feature_vectors);

  double learning_rate = 0.1;

  std::vector<int> gcn_embedding_dimensions;
  gcn_embedding_dimensions.push_back(feature_dim);
  gcn_embedding_dimensions.push_back(32);
  gcn_embedding_dimensions.push_back(3);

  std::vector<MatrixXf> layer_weights, layer_skip_weights, d_layer_weights, d_layer_skip_weights,
                      layer_weights_momentum, layer_weights_velocity, layer_skip_weights_momentum,
                      layer_skip_weights_velocity;

  for (int i = 0; i < gcn_embedding_dimensions.size()-1; i++) {
    layer_weights.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    d_layer_weights.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    d_layer_skip_weights.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_weights_momentum.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_weights_velocity.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights_velocity.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights_momentum.push_back(MatrixXf(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
  }

  for (int i = 0; i < layer_weights.size(); i++) {
    zero_init(layer_weights_momentum[i]);
    zero_init(layer_weights_velocity[i]);
    zero_init(layer_skip_weights_momentum[i]);
    zero_init(layer_skip_weights_velocity[i]);
  }

  std::default_random_engine generator(1000);
  for (int i = 0; i < layer_weights.size(); i++) {
    random_init(generator, layer_weights[i]);
    random_init(generator, layer_skip_weights[i]);
  }

  for (int64_t i = 0; i < GA.n; i++) {
    vIndices[i] = true;
  }
  vertexSubset Frontier(n, n, vIndices); //creates initial frontier

  for (int iter = 0; iter < 30; iter++) {
    std::vector<MatrixXf*> embedding_list, d_embedding_list, pre_embedding_list, d_pre_embedding_list;
    embedding_list.push_back(&(feature_vectors[0]));
    for (int i = 0; i < gcn_embedding_dimensions.size(); i++) {
      if (i > 0) embedding_list.push_back(new MatrixXf[GA.n]);
      pre_embedding_list.push_back(new MatrixXf[GA.n]);
      d_pre_embedding_list.push_back(new MatrixXf[GA.n]);
      d_embedding_list.push_back(new MatrixXf[GA.n]);
    }

    // Forward
    for (int i = 0; i < embedding_list.size()-1; i++) {
      MatrixXf* prev_vertex_embeddings = embedding_list[i];
      MatrixXf* next_vertex_embeddings = embedding_list[i+1];

      MatrixXf& weights = layer_weights[i];
      MatrixXf& skip_weights = layer_skip_weights[i];

      parallel_for (int j = 0; j < GA.n; j++) {
        next_vertex_embeddings[j] = MatrixXf(weights.rows(),1);
        pre_embedding_list[i][j] = MatrixXf(weights.rows(),1);
        zero_init(next_vertex_embeddings[j]);
        zero_init(pre_embedding_list[i][j]);
      }
      bool first = (i == embedding_list.size()-2);

      vertexMap(Frontier, GCN_applyweights_F<vertex>(Parents, GA, weights, pre_embedding_list[i],
                                        prev_vertex_embeddings));
      vertexMap(Frontier, GCN_F<vertex>(Parents, GA, weights, skip_weights, next_vertex_embeddings,
                                        pre_embedding_list[i], prev_vertex_embeddings, first));

      //edgeMap(GA, Frontier, GCN_edgeMap_F<vertex>(Parents, GA, weights, skip_weights, next_vertex_embeddings,
      //                                  pre_embedding_list[i], prev_vertex_embeddings, first));
    }

    MatrixXf* final_vertex_embeddings = new MatrixXf[GA.n];
    parallel_for (int i = 0; i < GA.n; i++) {
      final_vertex_embeddings[i] = MatrixXf(3,1);
      softmax(embedding_list[embedding_list.size()-1][i], final_vertex_embeddings[i]);
    }

    double* losses = new double[GA.n];
    double total_loss = 0.0;

    reducer_opadd<double> total_loss_reducer(total_loss);

    int batch_size = 0;
    int total_val_correct = 0;
    int total_val = 0;

    reducer_opadd<int> batch_size_reducer(batch_size);
    reducer_opadd<int> total_val_correct_reducer(total_val_correct);
    reducer_opadd<int> total_val_reducer(total_val);

    parallel_for (int i = 0; i < GA.n; i++) {
      if (!is_train[i]) {
        losses[i] = 0.0;
        if (is_val[i]) continue;
        crossentropy(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
        int argmax = 0;
        int gt_label = 0;
        double maxval = -1;
        for (int j = 0; j < final_vertex_embeddings[i].rows(); j++) {
          if (final_vertex_embeddings[i](j,0) > maxval || j == 0) {
            argmax = j;
            maxval = final_vertex_embeddings[i](j,0);
          }
          if (groundtruth_labels[i](j,0) > 0.5) gt_label = j;
        }
        if (gt_label == argmax) {
          *total_val_correct_reducer += 1;
        }
        *total_val_reducer += 1;
        continue;
      }
      *batch_size_reducer += 1;
      crossentropy(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
      *total_loss_reducer += losses[i];
    }
    total_loss = total_loss_reducer.get_value();
    batch_size = batch_size_reducer.get_value();
    total_val_correct = total_val_correct_reducer.get_value();
    total_val = total_val_reducer.get_value();

    printf("epoch %d: \ttotal loss is %f test accuracy %f\n", iter+1, total_loss/batch_size,
                                                              (1.0*total_val_correct) / total_val);

    // now do reverse.
    MatrixXf* d_final_vertex_embeddings = new MatrixXf[GA.n];
    parallel_for (int i = 0; i < GA.n; i++) {
      double d_loss = 1.0/batch_size;
      if (!is_train[i]) d_loss = 0.0;
      d_final_vertex_embeddings[i] = MatrixXf(final_vertex_embeddings[i].rows(),
                                              final_vertex_embeddings[i].cols());
      zero_init(d_final_vertex_embeddings[i]);
      d_crossentropy(final_vertex_embeddings[i], d_final_vertex_embeddings[i],
                     groundtruth_labels[i], d_loss);
    }

    MatrixXf* d_next_vertex_embeddings = d_embedding_list[d_embedding_list.size()-1];
    MatrixXf* next_vertex_embeddings = embedding_list[d_embedding_list.size()-1];
    parallel_for (int i = 0; i < GA.n; i++) {
      d_softmax(next_vertex_embeddings[i], d_next_vertex_embeddings[i],
                final_vertex_embeddings[i], d_final_vertex_embeddings[i]);
    }

    for (int i = embedding_list.size()-2; i >= 0; --i) {
      bool first = (i == embedding_list.size()-2);
      MatrixXf* d_weights = &(d_layer_weights[i]);
      MatrixXf* d_skip_weights = &(d_layer_skip_weights[i]);
      MatrixXf& weights = layer_weights[i];
      MatrixXf& skip_weights = layer_skip_weights[i];
      zero_init(*d_weights);
      zero_init(*d_skip_weights);

      MatrixXf* prev_vertex_embeddings = embedding_list[i];
      MatrixXf* next_vertex_embeddings = embedding_list[i+1];
      MatrixXf* d_prev_vertex_embeddings = d_embedding_list[i];
      MatrixXf* d_next_vertex_embeddings = d_embedding_list[i+1];

      parallel_for (int j = 0; j < GA.n; j++) {
        d_prev_vertex_embeddings[j] = MatrixXf(prev_vertex_embeddings[j].rows(),
                                               prev_vertex_embeddings[j].cols());
        d_pre_embedding_list[i][j] = MatrixXf(pre_embedding_list[i][j].rows(),
                                              pre_embedding_list[i][j].cols());
        zero_init(d_prev_vertex_embeddings[j]);
        zero_init(d_pre_embedding_list[i][j]);
      }

      ArrayReducer reducer;

      vertexMap(Frontier, d_GCN_F<vertex>(Parents, GA, weights, d_weights,
                                          skip_weights, d_skip_weights,
                                          next_vertex_embeddings, d_next_vertex_embeddings,
                                          pre_embedding_list[i], d_pre_embedding_list[i],
                                          prev_vertex_embeddings, d_prev_vertex_embeddings,
                                          &reducer, first));
      //edgeMap(GA, Frontier, d_GCN_edgeMap_F<vertex>(Parents, GA, weights, d_weights,
      //                                    skip_weights, d_skip_weights,
      //                                    next_vertex_embeddings, d_next_vertex_embeddings,
      //                                    pre_embedding_list[i], d_pre_embedding_list[i],
      //                                    prev_vertex_embeddings, d_prev_vertex_embeddings,
      //                                    &reducer, first), remove_duplicates);

      reducer.combine();
      {
        ArrayReducerView* view = reducer.view();
        parallel_for (int j = 0; j < GA.n; j++) {
          if (view->has_view(&(d_pre_embedding_list[i][j]))) {
            d_pre_embedding_list[i][j] = *(view->get_view(&(d_pre_embedding_list[i][j])));
            delete view->get_view(&(d_pre_embedding_list[i][j]));
          }
          //if (view->has_view(&(d_prev_vertex_embeddings[j]))) {
          //  //d_prev_vertex_embeddings[j] = *(view->get_view(&(d_prev_vertex_embeddings[j])));
          //  //delete view->get_view(&(d_prev_vertex_embeddings[j]));
          //}
        }
      }

      vertexMap(Frontier, d_GCN_applyweights_F<vertex>(Parents, GA, weights, d_weights,
                                          pre_embedding_list[i], d_pre_embedding_list[i],
                                          prev_vertex_embeddings, d_prev_vertex_embeddings,
                                          &reducer));
      reducer.combine();
      ArrayReducerView* view = reducer.view();
      parallel_for (int i = 0; i < GA.n; i++) {
        if (view->has_view(&(d_prev_vertex_embeddings[i]))) {
          d_prev_vertex_embeddings[i] = *(view->get_view(&(d_prev_vertex_embeddings[i])));
          delete view->get_view(&(d_prev_vertex_embeddings[i]));
        }
      }
      *d_weights = *(view->get_view(d_weights));
      *d_skip_weights = *(view->get_view(d_skip_weights));
      delete view->get_view(d_weights);
      delete view->get_view(d_skip_weights);
    }

    apply_gradient_update_ADAM(layer_weights, d_layer_weights, layer_weights_velocity,
                               layer_weights_momentum, 1.0, learning_rate, iter+1);
    apply_gradient_update_ADAM(layer_skip_weights, d_layer_skip_weights,
                               layer_skip_weights_velocity, layer_skip_weights_momentum, 1.0,
                               learning_rate, iter+1);

    for (int i = 0; i < embedding_list.size(); i++) {
      if (i > 0) delete[] embedding_list[i];

      delete[] d_embedding_list[i];
      delete[] pre_embedding_list[i];
      delete[] d_pre_embedding_list[i];
    }

    delete[] final_vertex_embeddings;
    delete[] losses;
    delete[] d_final_vertex_embeddings;
  }
  Frontier.del();

  free(is_train);
  free(is_val);
  free(is_test);
  free(Parents);
}
