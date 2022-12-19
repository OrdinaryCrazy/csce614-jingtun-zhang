#include <utility>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::pair;
using std::ios;

void parse_pubmed_data(std::string train_filename, std::string val_filename,
                       std::string test_filename, std::string feature_filename, bool* is_train,
                       bool* is_val, bool* is_test, std::vector<MatrixXf>& labels,
                       std::vector<MatrixXf>& features) {
  {
    std::ifstream f(train_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;

      if (c1+c2+c3 > 0.5) {
        is_train[id] = true;
        labels[id](0,0) = c1;
        labels[id](1,0) = c2;
        labels[id](2,0) = c3;
      }
    }
  }

  {
    std::ifstream f(val_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;
      //printf("%d, %f %f %f \n", id, c1, c2, c3);

      if (c1+c2+c3 > 0.5) {
        is_val[id] = true;
        labels[id](0,0) = c1;
        labels[id](1,0) = c2;
        labels[id](2,0) = c3;
      }
    }
  }

  {
    std::ifstream f(test_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;
      //printf("%d, %f %f %f \n", id, c1, c2, c3);

      if (c1+c2+c3 > 0.5) {
        is_test[id] = true;
        labels[id](0,0) = c1;
        labels[id](1,0) = c2;
        labels[id](2,0) = c3;
      }
    }
  }

  {
    std::ifstream f(feature_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id, fidx;
      float weight;
      l >> id >> fidx >> weight;
      //printf("%d %d %f\n", id, fidx, weight);
      features[id](fidx,0) = weight;
    }
  }
}


