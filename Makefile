GPU=1
CUDNN=1
OPENCV=0
DEBUG=0

ARCH= \
      -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52

VPATH=./darknet/src/
OBJDIR=./obj/

CC=gcc
GCC=g++
NVCC=nvcc 
OPTS=-Ofast -w
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-fPIC -Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
# OPTS=-O0 -g
OPTS= -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/ -I$(shell pwd)/darknet/src/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif
tn = pyolo

OBJ=gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o super.o voxel.o tree.o $(tn)c.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./darnket/src/*.h) Makefile
all: obj $(tn).so

CVFLAGS= `pkg-config --libs opencv` 
BOOSTFLAGS= -lboost_python -lboost_numpy
export PATH := /usr/local/cuda/bin:$(PATH)
$(tn).so: $(OBJS)
	g++ -g -w -I/usr/local/cuda/include/ -I/usr/include/python2.7 -I$(shell pwd)/darknet/src  -Wall -fPIC  -Ofast -c ./src/$(tn).cpp -o obj/$(tn).o
	$(GCC) -shared -Wl,-soname,$@ -o $@ $^ ./obj/$(tn).o $(BOOSTFLAGS) $(CVFLAGS)  $(LDFLAGS)


$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
	$(CC) $(COMMON) $(CFLAGS) -c ./src/$(tn)c.c -o ./obj/$(tn)c.o

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(tn).so  $(OBJDIR)$(tn).o
