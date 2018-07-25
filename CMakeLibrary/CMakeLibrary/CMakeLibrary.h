// CMakeLibrary.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#define MODULE_API_EXPORTS

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef MODULE_API_EXPORTS
#    define MODULE_API __declspec(dllexport)
#  else
#    define MODULE_API __declspec(dllimport)
#  endif
#else
#  define MODULE_API
#endif

	MODULE_API double Add(double x, double y);

#ifdef __cplusplus
}
#endif

// TODO: Reference additional headers your program requires here.
