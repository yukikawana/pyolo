#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>
extern "C" {
	#include <image.h>
}
extern "C" void cinit(const char* dc, const char* cf, const char* wf); 
extern "C" int cpredict1(image *im);
extern "C" void cpredict2(int arr[]);
namespace bp = boost::python;
namespace np = boost::numpy;
using namespace std;
void init(string dc, string cf, string wf)
{
	cinit(dc.c_str(), cf.c_str(), wf.c_str());
	}

np::ndarray predict(np::ndarray &bpim){
	
  printf("convert ");
	printf("%d\n", bpim.shape(0));
	printf("%d\n", bpim.shape(1));
	printf("%d\n", bpim.shape(2));
printf("convert done");
    int h = bpim.shape(0);
    int w = bpim.shape(1);
    int c = bpim.shape(2);
    image out = make_image(w, h, c);
    int i, j, k, count=0;;
    const long int* strides = bpim.get_strides();
    unsigned char* data = (unsigned char*)bpim.get_data();
    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
           out.data[count++] = ((float)*(data + k*strides[2]+ i * strides[0] +j * strides[1]))/255.;
           //int dst_index = j + w*i + w*h*k;
           //out.data[dst_index] = (float)*(bpim.get_data() + k*strides[2]+ i * strides[0] +j * strides[1])/255.;
            }
        }
    }
    
    printf("convert mage done");
    const int retlen=cpredict1(&out);
    int ret[retlen]={};
    cpredict2(ret); 
bp::tuple shapeB = bp::make_tuple(retlen*6);
np::ndarray b = np::zeros(shapeB, np::dtype::get_builtin<int>());
printf("ret len=%d",(int)retlen);
for(int i = 0; i< retlen*6; i++)
{  
	b[i] = ret[i];
	printf("a = %d\n",ret[i]);
	}
printf("ret address %p\n", ret);
//free(ret);
    printf("test1 done");
    return b;
}
BOOST_PYTHON_MODULE(pyolo){
np::initialize();
//boost::python::def( "testwrap1", testwrap1, bp::return_value_policy<bp::return_by_value>());
bp::def( "init", init);
bp::def( "predict", predict);
//boost::python::def( "test4wf", &test4wf,bp::return_value_policy<bp::reference_existing_object>());
}
