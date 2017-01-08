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
image im;
char *voc_names2[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char** names;
	float thresh = .24;
	box *boxes;
	float** probs;
void cinit(const char* dc, const char* cf, const char* wf)
{
gpu_index = 0;


    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
    list *options = read_data_cfg(dc);
    char *name_list = option_find_str(options, "names", "data/names.list");
    names = get_labels(name_list);

    net = parse_network_cfg(cf);
    if(wf){
        load_weights(&net, wf);
    }
    set_batch_network(&net, 1);
   l = net.layers[net.n-1];
    srand(2222222);

	}
int cpredict1(image *im1)
{
im = *im1;
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
        image sized = resize_image(im, net.w, net.h);
        boxes = calloc(l.w*l.h*l.n, sizeof(box));
        probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
int num = l.w*l.h*l.n;
int si=0;
for(int j = 0; j< num; j++)
{
  int class = max_index(probs[j], l.classes);
  float prob = probs[j][class];
	if(prob > thresh){
      si++;       
      printf("%s: %.0f%%\n", names[class], prob*100);
		}
	}
        free_image(sized);
	return si;
}
void cpredict2(int ret[]){
int si = 0;
int num = l.w*l.h*l.n;
    for(int i = 0; i < num; ++i){
            box b = boxes[i];
        int class = max_index(probs[i], l.classes);
        float prob = probs[i][class];
	if(prob > thresh){
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
        ret[si++] = class;
        ret[si++] = (int)100*prob;
        ret[si++] = left;
        ret[si++] = right;
        ret[si++] = top;
        ret[si++] = bot;
        printf("%s: %d %d %d %d %d\n", names[ret[si-6]], ret[si-5], ret[si-4], ret[si-3], ret[si-2], ret[si-1]);
    }
		}
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        free_image(im);

}
