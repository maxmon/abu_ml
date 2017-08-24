# 将$DATA文件下的数据文件转换成$OUTPUT下的train_lmdb和test_lmdb

# 工程目录
PORJECT_ENV=/root/maxmon/lr
# caffe安装目录
CAFFE_ENV=/root/caffe
    
OUTPUT=$PORJECT_ENV/gen 
DATA=$PORJECT_ENV/data
# 使用$CAFFE_ENV/build/examples/mnist/convert_mnist_data这个工具完成lmdb转换
BIN=$CAFFE_ENV/build/examples/mnist/convert_mnist_data

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

mkdir -p $OUTPUT

$BIN $DATA/train-images-idx3-ubyte $DATA/train-labels-idx1-ubyte $OUTPUT/train_lmdb --backend=${BACKEND}
$BIN $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte $OUTPUT/test_lmdb --backend=${BACKEND}
