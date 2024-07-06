# conv2d_direct

直接计算conv2d的cuda算子实现

详细介绍参考：https://zhuanlan.zhihu.com/p/613538649

## 编译

需要确保本机有cuda和cudnn

`nvcc main.cu -o test -lcudnn`

如果找不到cudnn请手动指定目录

## 测试

报告生成：`nsys profile --stats=true -o report_conv ./test `

ncu生成：`sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda-11.7/lib64  /path/to/cuda-11.7/bin/ncu test`

| N/batch_size | inC | inH  | inW  | outC | outH | outW | kernelH | kernelW | cudnn    | v1_conv  | speedup     |
|--------------|-----|------|------|------|------|------|---------|---------|----------|----------|-------------|
| 1            | 3   | 768  | 512  | 3    |      |      | 3       | 3       | 0.106353 | 0.119194 | 0.892268067 |
| 1            | 3   | 840  | 1200 | 3    |      |      | 3       | 3       | 0.29952  | 0.282706 | 1.059475215 |
| 1            | 3   | 960  | 1440 | 3    |      |      | 3       | 3       | 0.407859 | 0.406088 | 1.004361124 |
| 1            | 3   | 1200 | 1680 | 3    |      |      | 3       | 3       | 0.593111 | 0.547789 | 1.082736236 |
| 1            | 3   | 1440 | 1920 | 3    |      |      | 3       | 3       | 0.7454   | 0.745237 | 1.000218722 |
| 1            | 3   | 1920 | 2400 | 3    |      |      | 3       | 3       | 1.349059 | 1.243177 | 1.085170495 |
|              |     |      |      |      |      |      |         |         |          |          |             |
| 1            | 3   | 768  | 512  | 3    |      |      | 6       | 6       | 0.256932 | 0.160492 | 1.600902226 |
| 1            | 3   | 840  | 1200 | 3    |      |      | 6       | 6       | 0.667402 | 0.378142 | 1.764950733 |
| 1            | 3   | 960  | 1440 | 3    |      |      | 6       | 6       | 0.808141 | 0.448737 | 1.800923481 |
| 1            | 3   | 1200 | 1680 | 3    |      |      | 6       | 6       | 1.326254 | 0.742543 | 1.786097236 |
| 1            | 3   | 1440 | 1920 | 3    |      |      | 6       | 6       | 1.771069 | 0.970639 | 1.824642323 |
| 1            | 3   | 1920 | 2400 | 3    |      |      | 6       | 6       | 2.904648 | 1.582961 | 1.834946028 |