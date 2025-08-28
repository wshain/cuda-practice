# cuda-practice
### reduce 
v0.源代码
v1.shared_memory优化
v2.ward分支优化：warp不要有空格，尽量聚在一起
v3.bank优化：避免处理warp里面相邻元素，会产生bank conflict 
    bank冲突即不同线程访问同一个bank内的不同元素（当访问相同元素时会有广播，不会冲突）
v4.planA:最初load加载到shared mem时直接先进行一次加法（block之间）
  planB:（block）内部
v5.最后一个warp优化：在最后一个warp中不进行__syncthreads()同步操作
v6.展开for循环：降低for循环开销
v7.合理设置block数量：并不是越少越好
v8.shuffle寄存器优化，放在寄存器中实现，快

### sgemm
<img src="/static/sgemm_cpu.png"></img>

##### (M,K) * (K,N) = (M,N)
v0
v1