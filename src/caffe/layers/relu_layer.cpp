#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

/**********************
 * 1、标准线性整流函数 ReLU
 * 
 *   y = max(0, x)
 *   y = x （当x>0时)， y = 0 (当x<=0时)
 * 
 *   梯度计算
 *   diff_x = diff_y*1 (当x>0时), diff_x = diff_y*0 (当x<=0时) 
 * 
 * 2、带泄露线性整流 Leaky ReLU
 * 
 *   y = x (当x>0时), y = ax (当x<=0时,其中 0<a<1)
 * 
 *   梯度计算
 *   diff_x = diff_y*1 (当x>0时), diff_x = diff_y*a (当x<=0时) 
 * 
 * 总结:
 *      1、输入为正的时，不会产生梯度弥散，收敛速度快
 *      
 *  缺点:
 *      1、梯度弥散问题没有完全解决，输入为负时，出现梯度消散，非激活的输入值无法进行反向传播。
 *      2、此时需要合理设置学习率，降低神经元死亡的概率
 *  优点:
 *      1、解决了部分梯度弥散问题
 *      2、收敛速度更快
 * 
 *  在神经网络中，隐含层的激活函数，sigmoid、tanh, ReLU 最好选择ReLU
 **********************/ 

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  //negative_slope=0 为标准ReLU, negative_slope!=0为Leaky ReLU
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
