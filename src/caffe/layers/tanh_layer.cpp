// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

/***********************************************************************
 *  双曲正切 tanh
 *  
 *  表达式:
 *            y = tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)) = 2sigmoid(2x)-1
 * 
 *  梯度计算:
 *            diff_y = diff_x * (1-tahn(x)*tahn(x))
 * 
 * 
 *  总结:
 *        1、tanh的变化敏感区间较宽
 *        2、在原点附近，tanh跟y=x相近，当x较小时，可以简化计算
 *  缺点:  
 *        1、梯度弥散问题没有解决
 *  优点:
 *        1、解决了原点对称问题、比sigmoid更快
 **********************************************************************/ 

namespace caffe {

template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
