#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

/*
 * @breif 针对网络中某一层中一个top连接到多个下一层bottom的情况。需要在本层之后添加一个Split层。
 *        作用就把一个数据输入bottom, 拷贝为多个top，没个top与下一层bottom一一对应
 * @demo       
          在lenet网络 TEST阶段, ip2和label分别作为accuracy和loss层的输入
          layer {
			  name: "mnist"
			  type: "Data"
			  top: "data"
			  top: "label"
			  include {
				phase: TEST
			  }
			  transform_param {
				scale: 0.00390625
			  }
			  data_param {
				source: "examples/mnist/mnist_test_lmdb"
				batch_size: 100
				backend: LMDB
			  }
			}
			
			...
			
			layer {
			  name: "accuracy"
			  type: "Accuracy"
			  bottom: "ip2"
			  bottom: "label"
			  top: "accuracy"
			  include {
				phase: TEST
			  }
			}
			layer {
			  name: "loss"
			  type: "SoftmaxWithLoss"
			  bottom: "ip2"
			  bottom: "label"
			  top: "loss"
			}
			
          此时需要分裂为:
          
          layer {
			  name: "mnist"
			  type: "Data"
			  top: "data"
			  top: "label"
			  include {
				phase: TEST
			  }
			  transform_param {
				scale: 0.00390625
			  }
			  data_param {
				source: "examples/mnist/mnist_test_lmdb"
				batch_size: 100
				backend: LMDB
			  }
			}
			layer {
			  name: "label_mnist_1_split"
			  type: "Split"
			  bottom: "label"
			  top: "label_mnist_1_split_0"
			  top: "label_mnist_1_split_1"
			}
			
			...
			
			layer {
			  name: "ip2_ip2_0_split"
			  type: "Split"
			  bottom: "ip2"
			  top: "ip2_ip2_0_split_0"
			  top: "ip2_ip2_0_split_1"
			}
			layer {
			  name: "accuracy"
			  type: "Accuracy"
			  bottom: "ip2_ip2_0_split_0"
			  bottom: "label_mnist_1_split_0"
			  top: "accuracy"
			  include {
				phase: TEST
			  }
			}
			layer {
			  name: "loss"
			  type: "SoftmaxWithLoss"
			  bottom: "ip2_ip2_0_split_1"
			  bottom: "label_mnist_1_split_1"
			  top: "loss"
			}			
 *        
 * 
 */ 

namespace caffe {

/*
 * @brief   插入Split层
 * @param   param       原神经网络
 *          param_split 分裂后的神经网络
 */ 
void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layer();
  // blob名<->(层索引,top blob索引)
  map<string, pair<int, int> > blob_name_to_last_top_idx;
  // (层索引,bottom blob索引)<->(层索引,top blob索引)
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  // (层索引,top blob索引)<->下层bottom个数
  map<pair<int, int>, int> top_idx_to_bottom_count;
  // (层索引,top blob索引)<->loss_weight(损失权重)
  map<pair<int, int>, float> top_idx_to_loss_weight;
  // (层索引,top blob索引)<->split id
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  // 层索引<->层名称
  map<int, string> layer_idx_to_layer_name;
  // 遍历网络
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    // 遍历层输入
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
	  // 获取输入blob name
      const string& blob_name = layer_param.bottom(j);
      // 本层的输入就是上层的输出，因此可用blob name做验证
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      // (当前层索引， bottom blob索引)
      const pair<int, int>& bottom_idx = make_pair(i, j);
      // 通过blob名字，获取(上一层索引，top blob索引)
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      // 当前层输入与上层输出对应关系
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      // bottom blob计数
      ++top_idx_to_bottom_count[top_idx];
    }
    // 遍历层输出
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as a bottom blob to another layer.
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const string& blob_name = layer_param.top(j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      // 保存损失权重
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      // 当损失权重不为零，bottom blob计数加一
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }
  // 遍历网络
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));
    // Replace any shared bottom blobs with split layer outputs.
    // 遍历层输入
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      // 获取(上层索引，top blob索引）		
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];
      // 获取 bottom blob计数  
      const int split_count = top_idx_to_bottom_count[top_idx];
      // bottom blob计数大于1时
      if (split_count > 1) {
		// 获取上层名称  
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];
        // 获取bottom blob名称
        const string& blob_name = layer_param->bottom(j);
        // bottom blob重命名
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    // 遍历输出层
    for (int j = 0; j < layer_param->top_size(); ++j) {
	  // 获取 （层索引，top blob索引
      const pair<int, int>& top_idx = make_pair(i, j);
      // 获取 bottom blob计数
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
		// 获取层名称
        const string& layer_name = layer_idx_to_layer_name[i];
        // 获取top blob名称
        const string& blob_name = layer_param->top(j);
        
        LayerParameter* split_layer_param = param_split->add_layer();
        // 获取层损失权重
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        // 新建Split层，添加top和weight
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

/*
 * @brief   创建Split层
 * @param   layer_name        分裂blob所处的层名
 *          blob_name         要分裂的blob名
 *          blob_idx          blob索引
 *          split_count       分裂索引
 *          loss_weight       损失函数权重
 *          split_layer_param Split层参数指针
 */
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  // 修改层名称
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  // 设置层类型
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    // 添加输出blob  
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    // 添加损失权重    
    if (loss_weight) {
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
  }
}

string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
