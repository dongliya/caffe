#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  //线程初始化
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  //线程停止
  void StopInternalThread();
  //线程是否启动
  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  //线程业务函数, 继承该类必须实现的函数
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  //请求退出前调用，查看是否已经处于中断请求状态
  bool must_stop();

 private:
  //线程要执行的函数
  void entry(int device, Caffe::Brew mode, int rand_seed,
      int solver_count, int solver_rank, bool multiprocess);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
