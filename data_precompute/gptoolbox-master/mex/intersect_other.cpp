
#include <igl/read_triangle_mesh.h>
#include <igl/pathinfo.h>
#include <igl/writeOFF.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/writeDMAT.h>
#include <igl/unique_simplices.h>
#include <igl/C_STR.h>
#define IGL_REDRUM_NOOP
#include <igl/REDRUM.h>
#include <igl/matlab/MexStream.h>
#include <igl/matlab/mexErrMsgTxt.h>
#include <igl/matlab/validate_arg.h>
#include <igl/matlab/parse_rhs.h>
#include <igl/matlab/prepare_lhs.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>

#include "mex.h"

#include <iostream>
#include <string>

#include <mex.h>
#undef assert
#define assert( isOK ) ( (isOK) ? (void)0 : (void) mexErrMsgTxt(C_STR(__FILE__<<":"<<__LINE__<<": failed assertion `"<<#isOK<<"'"<<std::endl) ) )

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
  // This is useful for debugging whether Matlab is caching the mex binary
  //mexPrintf("%s %s\n",__TIME__,__DATE__);
  igl::matlab::MexStream mout;
  std::streambuf *outbuf = std::cout.rdbuf(&mout);

  using namespace std;
  using namespace Eigen;
  using namespace igl;
  using namespace igl::matlab;
  using namespace igl::copyleft::cgal;
  igl::copyleft::cgal::RemeshSelfIntersectionsParam params = {nlhs<=1,false};

  MatrixXd VA,VB;
  MatrixXi FA,FB;

  string prefix;
  bool use_obj_format = false;
  const int NUM_REQ = 4;
  if(nrhs < NUM_REQ)
  {
    mexErrMsgTxt(C_STR("nrhs < "<<NUM_REQ));
  }
  const auto & parse_mesh = [](
      const mxArray *prhs[],
      MatrixXd & V,
      MatrixXi & F)
  {
    parse_rhs_double(prhs,V);
    parse_rhs_index(prhs+1,F);
    mexErrMsgTxt(V.cols()==3,"V must be #V by 3");
    mexErrMsgTxt(F.cols()==3,"F must be #F by 3");
  };
  parse_mesh(prhs,VA,FA);
  parse_mesh(prhs+2,VB,FB);

  if(nrhs>NUM_REQ)
  {
    int i = NUM_REQ;
    while(i<nrhs)
    {
      if(!mxIsChar(prhs[i]))
      {
        mexErrMsgTxt("Parameter names should be char strings");
      }
      // Cast to char
      const char * name = mxArrayToString(prhs[i]);
      if(strcmp("FirstOnly",name) == 0)
      {
        validate_arg_logical(i,nrhs,prhs,name);
        validate_arg_scalar(i,nrhs,prhs,name);
        mxLogical * v = (mxLogical *)mxGetData(prhs[++i]);
        params.first_only = *v;
      }else
      {
        mexErrMsgTxt(C_STR("Unsupported parameter: "<<name));
      }
      i++;
    }
  }
  const auto validate = [](const MatrixXd & V, const MatrixXi & F) -> bool
  {
    // Check that there aren't any combinatorially or geometrically degenerate triangles
    VectorXd A;
    doublearea(V,F,A);
    if(A.minCoeff()<=0)
    {
      mexErrMsgTxt("Geometrically degenerate face found.");
    }
    VectorXi F12,F23,F31;
    F12 = F.col(0)-F.col(1);
    F23 = F.col(1)-F.col(2);
    F31 = F.col(2)-F.col(0);
    if(
      F12.minCoeff() == 0 || 
      F23.minCoeff() == 0 || 
      F31.minCoeff() == 0)
    {
      mexErrMsgTxt("Combinatorially degenerate face found.");
    }
    return true;
  };

  if(!validate(VA,FA) || !validate(VB,FB))
  {
    // should have already called mexErr
    return;
  }

  // Now mesh self intersections
  Eigen::MatrixXd VVAB;
  Eigen::MatrixXi FFAB,IF;
  Eigen::VectorXi JAB,IMAB;
  {
    igl::copyleft::cgal::intersect_other(
      VA,FA,VB,FB,params,IF,VVAB,FFAB,JAB,IMAB);
  }
  switch(nlhs)
  {
    default:
    {
      mexErrMsgTxt(false,"Too many output parameters.");
    }

    case 5:
    {
      prepare_lhs_index(IMAB,plhs+4);
      // Fall through
    }
    case 4:
    {
      prepare_lhs_index(JAB,plhs+3);
      // Fall through
    }
    case 3:
    {
      prepare_lhs_index(FFAB,plhs+2);
      // Fall through
    }
    case 2:
    {
      prepare_lhs_double(VVAB,plhs+1);
      // Fall through
    }

    case 1:
    {
      prepare_lhs_index(IF,plhs+0);
      // Fall through
    }
    case 0: break;
  }

  // Restore the std stream buffer Important!
  std::cout.rdbuf(outbuf);

}
