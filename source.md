https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/82634878

pip install tensorflow-gpu==2.0.0b1  -i https://pypi.tuna.tsinghua.edu.cn/simple



in pytorch,
loss.backward()------->
torch.autograd.backward(self, gradient, retain_graph, create_graph)(location: python2.7/site-packages/torch/tensor.py  class Tensor.backward(self, gradient=None, retain_graph=None, create_graph=False))------->
tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)------->
_make_grads(outputs, grads) --> return tuple(new_grads)(outputs is loss, tensor(2.3658, device='cuda:0', grad_fn=<NllLossBackward>) )------->
     Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True) 
  ( 
  location python2.7/site-packages/torch/autograd/Variable
  from torch._C import _ImperativeEngine as ImperativeEngine
  Variable._execution_engine = ImperativeEngine()
  )
  
  ---------------------------------------------------------------------------------------
  GradientDescentOptimizer------->
  minize(loss)------->
  class Optimizer(
    # Optimizers inherit from CheckpointableBase rather than Checkpointable
    # since they do most of their dependency management themselves (slot
    # variables are special-cased, and non-slot variables are keyed to graphs).
    checkpointable.CheckpointableBase)
  --> 
    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None)
  -->     grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)
        
  ------->_get_processor(v):
  
  ------->_RefVariableProcessor(v)
  
  
  
  ------->grads = gradients.gradients(
        loss, var_refs, grad_ys=grad_loss,
        gate_gradients=(gate_gradients == Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
        
        
*************************************classification tf slim*************************************
  
https://github.com/Robinatp/Tensorflow_Model_Slim_Classify
  
  clone source code
  git clone https://github.com/Robinatp/Tensorflow_Model_Slim_Classify.git
  
  download data and training
 
  cd Tensorflow_Model_Slim_Classify
 ./scripts/train_cifarnet_on_cifar10.sh
 
 
  https://github.com/PanJinquan/tensorflow_models_learning
  训练修改 resnet_v1_train_val.py
  
  预测使用 predict.py
  *************************************classification tf slim*************************************
  
  
  ## CUDA_ERROR_NO_DEVICE
  python main.py --phase train --dataset tiny --res_n 18 --lr 0.1
Using TensorFlow backend.
2019-05-17 15:49:11.762192: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-05-17 15:49:12.675867: E tensorflow/stream_executor/cuda/cuda_driver.cc:397] failed call to cuInit: CUDA_ERROR_NO_DEVICE
2019-05-17 15:49:12.675949: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: fpga07
2019-05-17 15:49:12.675955: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: fpga07
2019-05-17 15:49:12.676017: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: 384.81.0
2019-05-17 15:49:12.676047: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: 384.81.0
2019-05-17 15:49:12.676052: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:300] kernel version seems to match DSO: 384.81.0

because set the environments CUDA_VISIBLE_DEVICES=-1, this setting will disable the nvidia device.

https://github.com/rasbt/deeplearning-models

  
