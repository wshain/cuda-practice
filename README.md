# cuda-practice
### reduce 归约求和
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

### sgemm 单精度通用矩阵乘法
##### (M,K) * (K,N) = (M,N)
<img src="/static/sgemm_cpu.png"></img>

#### read：2MNK 
#### write：MN

<font color = "red">v0</font>:gloabl memory 直接计算

-----------------------------------------

<img src ="/static/shared_memory.png">


<font color = "red">v1</font>: shared memory 先分别将两个矩阵中的一长块（整体）（长K宽block）放入shared memory再进行计算

-------------------------------------

<i>上面的问题是一般shared memory 大小有限制，很难进行大矩阵的放入</i>
<font color = "red">v2</font>:shared memory sliding window 相当于滑动窗口 将矩阵中的一长块 按block大小放入shared memory 进行 计算，这里有一个前提就是，矩阵乘法计算中，直接相乘相加计算结果矩阵的每一个元素与分块相乘计算块，再相加块都可以计算出矩阵的乘法，等价的。
<img src="/static/shared_memory_sliding_window.png"></img>

------------------------------------

<font color = "red">v3</font>:my_sgemm_v3_increase_work_of_per_thread， 计算量太少，无法掩盖访存带宽，增加计算访存比,一次访存多几次计算
<img src= "/static/sgemm_v3_increase_work_of_per_thread.png">