#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "image.h"
#include "parser.h"
#include "cuda.h"
#include "blas.h"
#include "connected_layer.h"
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "box.h"
#include "demo.h"

network net;
layer l;
char** names;
float thresh = .24;
box *boxes;
float** probs;
int image_width;
int image_hight;

void cinit(const char* label_info_file, const char* net_structure_file, const char* weight){
	gpu_index = 0;

	if(gpu_index >= 0){
		cuda_set_device(gpu_index);
	}
	//setting up the net
	list *options = read_data_cfg(label_info_file);
	char *name_list = option_find_str(options, "names", "data/names.list");
	names = get_labels(name_list);

	net = parse_network_cfg(net_structure_file);
	if(weight){
		load_weights(&net, weight);
	}
	set_batch_network(&net, 1);
	l = net.layers[net.n-1];
	
	//prepare arrays for detected object info
	srand(2222222);
	boxes = calloc(l.w*l.h*l.n, sizeof(box));//array for the coordinates of the position of the detected object
	probs = calloc(l.w*l.h*l.n, sizeof(float *));// array for the confidence for each detected object
	for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
}

int get_number_of_objects_in_image(image *im){
	image_width = im->w;
	image_hight = im->h;
	clock_t time;
	char buff[256];
	char *input = buff;
	float nms=.4;

	image sized = resize_image(*im, net.w, net.h);//the net can only take the fixed size image. so the given image with the varying size has to be adjusted.

	float *X = sized.data;
	time=clock();
	network_predict(net, X);
	printf("Predicted in %f seconds.\n", sec(clock()-time));
	free_image(sized);
	get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
	if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);// non maxima supression

	int num = l.w*l.h*l.n;//total number of detection candidates
	int number_of_objects=0;

	for(int j = 0; j< num; j++){
	int class = max_index(probs[j], l.classes);
	float prob = probs[j][class];

	if(prob > thresh){// thresholding the candidates to choose the detection result with confidence larger than the threshold
		number_of_objects++;
		printf("%s: %.0f%%\n", names[class], prob*100);
		}
	}
	return number_of_objects;
}
void get_object_info(int ret[]){
	int number_of_objects = 0;
	int num = l.w*l.h*l.n;

	for(int i = 0; i < num; ++i){
		box b = boxes[i];
		int class = max_index(probs[i], l.classes);//decide which class the object belongs to
		float prob = probs[i][class];//how confident is it?
		if(prob > thresh){
			int left	= (b.x-b.w/2.)*image_width;
			int right = (b.x+b.w/2.)*image_width;
			int top	 = (b.y-b.h/2.)*image_hight;
			int bot	 = (b.y+b.h/2.)*image_hight;
			ret[number_of_objects++] = class;
			ret[number_of_objects++] = (int)(100*prob);
			ret[number_of_objects++] = left;
			ret[number_of_objects++] = right;
			ret[number_of_objects++] = top;
			ret[number_of_objects++] = bot;
		}
	}
}
