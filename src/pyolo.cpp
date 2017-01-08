#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>
#include <pthread.h>
extern "C" {
#include <image.h>
}
extern "C" void cinit(const char* dc, const char* cf, const char* wf); 
extern "C" int cpredict1(image *im);
extern "C" void cpredict2(int arr[]);
namespace bp = boost::python;
namespace np = boost::numpy; 
using namespace std;
pthread_mutex_t mutex2;
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
  //pthread_mutex_lock(&mutex2);   
  image out = make_image(w, h, c);
  printf("mage out");
  int i, j, k, count=0;;
  const long int* strides = bpim.get_strides();
  unsigned char* data = (unsigned char*)bpim.get_data();
  const int rgb[3]={2,1,0};
  for(k= 0; k < c; ++k){
    for(i = 0; i < h; ++i){
      for(j = 0; j < w; ++j){
out.data[count++] = ((float)*(data + rgb[k]*strides[2]+ i * strides[0] +j * strides[1]))/255.;
      }
    }
  }
  printf("convert image done");
  int retlen=cpredict1(&out);
 if(retlen>0){
      int ret[retlen*6]={0};
      cpredict2(ret);
      printf("retlen = %d\n", retlen);
      np::ndarray b = np::zeros(bp::make_tuple(retlen*6), np::dtype::get_builtin<int>());
      printf("nd %d\n",b.get_nd());
  const long int* bs= b.get_strides();
  int* data = (int*)b.get_data();
      printf("define b\n");
      printf("ret len=%d",(int)retlen);
      for(int i = 0; i< retlen*6; i++)
	{  
	  //*(data+i*bs[0]) = ret[i];
	  data[i] = ret[i];
	  if(i%6==0)printf(":::::::::::\n");
	  printf("a = %d\n",ret[i]);
	}
      printf("ret address %p\n", ret);
      printf("test1 done\n"); 
     free_image(out);
  return b;
	      } 
	      else{
      np::ndarray b = np::zeros(bp::make_tuple(1), np::dtype::get_builtin<int>());
  //pthread_mutex_unlock(&mutex2);   
     free_image(out);
      return b;
		      }
}
BOOST_PYTHON_MODULE(pyolo){
  np::initialize();
  //boost::python::def( "testwrap1", testwrap1, bp::return_value_policy<bp::return_by_value>());
  bp::def( "init", init);
  bp::def( "predict", predict);
  //boost::python::def( "test4wf", &test4wf,bp::return_value_policy<bp::reference_existing_object>());
}
