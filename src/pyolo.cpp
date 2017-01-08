#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>
#include <pthread.h>

extern "C"{
#include <image.h>
void cinit(const char* label_info_file, const char* net_structure_file, const char* weight);
int get_number_of_objects_in_image(image *im);
void get_object_info(int arr[]);
}

namespace bp = boost::python;
namespace np = boost::numpy; 
using namespace std;

void init(string label_info_file, string net_structure_file, string weight){
	printf("initialize the net");
	cinit(label_info_file.c_str(), net_structure_file.c_str(), weight.c_str());
	printf("initialization done");
}

np::ndarray predict(np::ndarray &bpim){
	int h = bpim.shape(0);
	int w = bpim.shape(1);
	int c = bpim.shape(2);
	
	image out = make_image(w, h, c);
	
	int i, j, k, count=0;;
	
	const long int* strides = bpim.get_strides();
	unsigned char* data = (unsigned char*)bpim.get_data();
	
	for(k= 2; k > -1; --k){//give image channels are in BGR order, so here it flips to RGB order.
		for(i = 0; i < h; ++i){
			for(j = 0; j < w; ++j){
				out.data[count++] = ((float)*(data + k*strides[2]+ i * strides[0] +j * strides[1]))/255.;//convert image data from boost ndarray type to darknet image type.
			}
		}
	}
	
	int number_of_objects=get_number_of_objects_in_image(&out);//see how many objects are in the given image.
	if(number_of_objects>0){
		int result[number_of_objects*6]={0};
		get_object_info(result);//get class, confidence, coordinates on the image for each object in the image. array is in the order below:
		//[class id][confidence][left][right][top][bottom]...
		printf("number_of_objects = %d\n", number_of_objects);

		np::ndarray b = np::zeros(bp::make_tuple(number_of_objects*6), np::dtype::get_builtin<int>());
		const long int* bs= b.get_strides();
		int* data = (int*)b.get_data();

		for(int i = 0; i< number_of_objects*6; i++){
			data[i] = result[i];//convert darknet image type to boost ndarray type.
		}

		free_image(out);

		return b;
	}
	else{//if no object is detected in the image, then simply returns empty ndarray
		np::ndarray b = np::zeros(bp::make_tuple(0), np::dtype::get_builtin<int>());
		free_image(out);
		return b;
	}
}

BOOST_PYTHON_MODULE(pyolo){
	np::initialize();
	bp::def( "init", init);
	bp::def( "predict", predict);
}
