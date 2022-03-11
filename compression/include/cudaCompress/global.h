#ifndef __TUM3D_CUDACOMPRESS__GLOBAL_H__
#define __TUM3D_CUDACOMPRESS__GLOBAL_H__


namespace cudaCompress {

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef long long int int64;
typedef unsigned long long int uint64;

}


#ifdef _WIN32
#define CUCOMP_EXPORT __declspec(dllexport)
#define CUCOMP_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define CUCOMP_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define CUCOMP_EXPORT
#endif // defined(__GNUC__)
#define CUCOMP_IMPORT CUCOMP_EXPORT
#endif // _WIN32

#ifdef RENDERER_BUILD_SHARED
#ifdef BUILD_MAIN_LIB
#define CUCOMP_DLL CUCOMP_EXPORT
#else
#define CUCOMP_DLL CUCOMP_IMPORT
#endif
#else 
#define CUCOMP_DLL
#endif


#endif
