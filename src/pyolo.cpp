#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>
#include <pthread.h>
#include <time.h>

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
	#ifdef TIME
		clock_t start = clock();
	#endif
	for(k= 2; k > -1; --k){//give image channels are in BGR order, so here it flips to RGB order.
		for(i = 0; i < h; ++i){
			for(j = 0; j < w; ++j){
				out.data[count++] = ((float)(data[ k*strides[2]+ i * strides[0] +j * strides[1]]))/255.;//convert image data from boost ndarray type to darknet image type.
			}
		}
	}
	#ifdef TIME
		clock_t end = clock();
		printf("# copy input image to darknet image format takes %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	#endif
	
	#ifdef TIME
		start = clock();
	#endif
	int number_of_objects=get_number_of_objects_in_image(&out);//see how many objects are in the given image.
	#ifdef TIME
		end = clock();
		printf("# to get number of objects in a image takes %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	#endif
	if(number_of_objects>0){
		int result[number_of_objects*6]={0};
	#ifdef TIME
		start = clock();
	#endif
		get_object_info(result);//get class, confidence, coordinates on the image for each object in the image. array is in the order below:
	#ifdef TIME
		end = clock();
		printf("# to get object info takes %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	#endif
		//[class id][confidence][left][right][top][bottom]...

		np::ndarray b = np::zeros(bp::make_tuple(number_of_objects,6), np::dtype::get_builtin<int>());
		const long int* bs= b.get_strides();
		int* data = (int*)b.get_data();

	#ifdef TIME
		start = clock();
	#endif
		for(int i = 0; i< number_of_objects; i++){
			for(int j = 0; j< 6; j++){
				b[i][j] = result[i*6+j];//convert darknet image type to boost ndarray type.
			}
		}
	#ifdef TIME
		end = clock();
		printf("# make array with result takes %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	#endif


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
