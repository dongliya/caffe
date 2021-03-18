#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

/**********************************************************************
 * Sigmoid函数，S型函数，S型生长曲线
 * 
 * 表达式:
 *        y = (1 + exp(-x))^{-1} = 0.5*tanh(0.5*x)+0.5
 * 
 * 梯度计算：
 *        diff_x = diff_y*y(1-y)
 * 
 * 总结:
 * 
 * 使用sigmoid作为激活函数时，随着神经网络隐含层数的增加，训练误差反而增大，容易引起梯度弥散（vanishing gradient），
 * 表现为:
 *    1、靠近输出层的隐含层梯度较大，参数更新速度快，很快就会收敛
 *    2、靠近输入层的隐含层梯度较小，参数更新速度慢，几乎和初始状态一样，随机分布
 * 
 * 缺点：
 *    1、容易梯度消失，限制网络层数，敏感区窄
 *    2、计算量较大，涉及除法
 *    3、不是关于原点对称，收敛速度慢
 * 
 **********************************************************************/ 

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
