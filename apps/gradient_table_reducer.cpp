
class ArrayReducerView {
  public:
    std::map<MatrixXf*, MatrixXf*> views;

    ArrayReducerView() { views.clear(); }

    void accumulate(ArrayReducerView* right) {
      for (auto iter = right->views.begin(); iter != right->views.end(); ++iter) {
        if (this->views.find(iter->first) == this->views.end()) {
          this->views[iter->first] = new MatrixXf(iter->second->rows(), iter->second->cols());
          *(this->views[iter->first]) = *(iter->second);
        } else {
          *(this->views[iter->first]) += *(iter->second);
        }
        delete iter->second;
      }
    }

    bool has_view(MatrixXf* id) {
      return this->views.find(id) != this->views.end();
    }

    MatrixXf* get_view(MatrixXf* id) {
      if (this->views.find(id) == this->views.end()) {
        // need to create a new version of this Matrix view.
        this->views[id] = new MatrixXf(id->rows(), id->cols());
        zero_init(*(this->views[id]));
      }
      return this->views[id];
    }
};


//struct _ArrayReducer : cilk::monoid_base<ArrayReducerView*> {
//  public:
//  static void reduce (ArrayReducerView** left, ArrayReducerView** right) {
//    (*left)->accumulate(*(right));
//  }
//
//  static void identity (ArrayReducerView** p) {
//    *p = new ArrayReducerView();
//  }
//};


ArrayReducerView* tls_array_reducer_view[256];


struct _tls_ArrayReducer {
  public:
    _tls_ArrayReducer() {
      for (int i = 0; i < 256; i++) {
        tls_array_reducer_view[i] = new ArrayReducerView();
      }
    }
    void combine() {
      int my_worker_id = getWorkerNum();//__cilkrts_get_worker_number();
      for (int i = 0; i < 256; i++) {
        if (i == my_worker_id) continue;
        tls_array_reducer_view[my_worker_id]->accumulate(tls_array_reducer_view[i]);
        delete tls_array_reducer_view[i];
        tls_array_reducer_view[i] = new ArrayReducerView();
      }
    }

    ArrayReducerView* view() {
      return tls_array_reducer_view[getWorkerNum()];
    }
};


template<typename T>
class reducer_opadd {
  public:
    T* wls;
    reducer_opadd(T val) {
      wls = (T*) calloc(getWorkers(), sizeof(T));
      wls[getWorkerNum()] = val;
    }

    T& operator* () {
      return wls[getWorkerNum()];
    }

    void combine() {
      int wid = getWorkerNum();
      for (int i = 0; i < getWorkers(); i++) {
        if (i == wid) continue;
        wls[wid] += wls[i];
      }
    }

    T get_value() {
      combine();
      return wls[getWorkerNum()];
    }
};




//typedef cilk::reducer<_ArrayReducer> ArrayReducer;
typedef _tls_ArrayReducer ArrayReducer;


